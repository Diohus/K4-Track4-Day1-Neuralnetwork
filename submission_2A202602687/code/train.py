"""One training pipeline; selection uses validation only."""
import copy
import random
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer

DEFAULT_CFG = dict(exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce", optimizer="sgd_momentum", lr=None, weight_decay=0.0, momentum=0.9,
    betas=(0.9, 0.999), eps=1e-8, batch=512, epochs=20, hidden=(256,128),
    dropout=0.0, init="he", clip_norm=None, precision="fp32", seed=1)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def macro_f1_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.float64)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.divide(2*np.diag(cm), denom, out=np.zeros(7), where=denom > 0).mean())

def compute_loss(logits, y, loss_name):
    if loss_name == "ce": return F.cross_entropy(logits, y)
    # MSE is mean over N*7 elements of RAW logits versus one-hot, not probabilities.
    if loss_name == "mse": return F.mse_loss(logits.float(), F.one_hot(y, 7).float())
    raise ValueError(f"Unknown loss: {loss_name}")

@torch.no_grad()
def predict(model, X, batch_size=8192):
    model.eval()
    return torch.cat([model(X[i:i+batch_size]).argmax(1) for i in range(0,len(X),batch_size)])

@torch.no_grad()
def evaluate(model, X, y, loss_name="ce", batch_size=8192):
    model.eval()
    total = torch.zeros((), device=X.device)
    cm = torch.zeros(49, dtype=torch.int64, device=X.device)
    for xb, yb in iterate_batches(X,y,batch_size,shuffle=False):
        logits = model(xb)
        total += compute_loss(logits,yb,loss_name)*len(yb)
        cm += torch.bincount(yb*7+logits.argmax(1),minlength=49)
    cm = cm.reshape(7,7).cpu().numpy()
    return dict(loss=float(total)/len(y), acc=float(np.trace(cm)/cm.sum()),
                macro_f1=macro_f1_from_confusion(cm))

def _sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)

def run_experiment(cfg, data):
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_tr"].device
    if cfg["precision"] != "fp32" and device.type != "cuda":
        raise ValueError("This AMP comparison requires CUDA; keep CPU baseline FP32")
    if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("GPU has no native BF16 support")
    set_seed(cfg["seed"])
    model = MLP(cfg["hidden"],cfg["dropout"],cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[tuple(cfg["hidden"])]
    opt = build_optimizer(cfg["optimizer"],model.parameters(),cfg["lr"],cfg["weight_decay"],
                          cfg["momentum"],tuple(cfg["betas"]),cfg["eps"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    step0 = evaluate(model,data["X_val"],data["y_val"],cfg["loss"])
    stats = activation_stats(model,data["X_val"][:2048])
    history = {k: [] for k in ("epoch","train_loss","val_loss","val_acc","val_macro_f1",
                               "grad_norm","grad_norm_max","clip_fraction","epoch_time_s")}
    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    best_loss, best_epoch, diverged = float("inf"), 0, False
    failure = ""
    # Fixed first 50k rows of the fixed stratified split, identical across runs.
    Xt, yt = data["X_tr"][:50000], data["y_tr"][:50000]
    for epoch in range(1,cfg["epochs"]+1):
        _sync(device); start=time.perf_counter(); model.train(); norms=[]
        for xb,yb in iterate_batches(data["X_tr"],data["y_tr"],cfg["batch"],gen):
            opt.zero_grad(set_to_none=True)
            context = nullcontext() if cfg["precision"] == "fp32" else torch.autocast(
                "cuda",dtype=torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16)
            with context: loss=compute_loss(model(xb),yb,cfg["loss"])
            if not torch.isfinite(loss):
                diverged=True; failure="non-finite loss"; break
            scaler.scale(loss).backward()
            # Unscale even without clipping: logged gradient must have its true scale.
            scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),
                float("inf") if cfg["clip_norm"] is None else cfg["clip_norm"])
            if not torch.isfinite(norm):
                if cfg["precision"] == "fp16":
                    scaler.step(opt); scaler.update(); continue
                diverged=True; failure="non-finite gradient"; break
            norms.append(norm.detach())
            scaler.step(opt); scaler.update()
        if diverged: break
        tr=evaluate(model,Xt,yt,cfg["loss"])
        va=evaluate(model,data["X_val"],data["y_val"],cfg["loss"])
        if not np.isfinite(tr["loss"]) or not np.isfinite(va["loss"]):
            diverged=True; failure="non-finite epoch evaluation"; break
        _sync(device); seconds=time.perf_counter()-start
        gn=torch.stack(norms) if norms else torch.zeros(1,device=device)
        values=dict(epoch=epoch,train_loss=tr["loss"],val_loss=va["loss"],val_acc=va["acc"],
                    val_macro_f1=va["macro_f1"],grad_norm=float(gn.mean()),grad_norm_max=float(gn.max()),
                    clip_fraction=float((gn>cfg["clip_norm"]).float().mean()) if cfg["clip_norm"] is not None else 0.,
                    epoch_time_s=seconds)
        for k,v in values.items(): history[k].append(v)
        if va["loss"] < best_loss:
            best_loss,best_epoch=va["loss"],epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"{cfg['exp_id']} {epoch:02d}/{cfg['epochs']} loss={va['loss']:.4f} "
              f"acc={va['acc']:.4f} f1={va['macro_f1']:.4f} grad={values['grad_norm']:.3f} {seconds:.2f}s",flush=True)
    index=best_epoch-1
    summary=dict(step0_loss=step0["loss"],best_val_loss=None if best_epoch==0 else best_loss,
        best_epoch=best_epoch,final_train_loss=history["train_loss"][-1] if history["epoch"] else None,
        final_val_loss=history["val_loss"][-1] if history["epoch"] else None,
        val_acc=history["val_acc"][index] if best_epoch else step0["acc"],
        val_macro_f1=history["val_macro_f1"][index] if best_epoch else step0["macro_f1"],
        time_per_epoch_s=float(np.mean(history["epoch_time_s"])) if history["epoch"] else None,
        peak_mem_MB=torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else None,
        diverged=diverged,completed_epochs=len(history["epoch"]),failure_reason=failure,
        activation_std=stats,train_metric_samples=len(Xt),device=str(device))
    return dict(cfg=cfg,history=history,summary=summary,best_state=best_state)

def write_predictions(row_id,preds,path):
    ids=np.asarray(row_id); preds=np.asarray(preds)
    if len(ids)!=len(preds) or len(np.unique(ids))!=len(ids): raise ValueError("Invalid row IDs")
    if not np.array_equal(preds,preds.astype(np.int64)) or ((preds<0)|(preds>6)).any():
        raise ValueError("Predictions must be integers 0..6")
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(row_id=ids.astype(np.int64),pred=preds.astype(np.int64))).to_csv(path,index=False)

def final_eval(cfg,result,data,pred_path):
    model=MLP(cfg["hidden"],cfg["dropout"],cfg["init"]).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    write_predictions(data["eval_row_id"],predict(model,data["X_eval"]).cpu().numpy(),pred_path)
