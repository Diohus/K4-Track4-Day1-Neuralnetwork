"""Reproducible Part 0-4 workflow. Checkpoints stay outside the submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from data import prepare_data
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from train import DEFAULT_CFG, set_seed, evaluate, compute_loss, run_experiment, final_eval
from plots import plot_run, plot_compare
from results_table import save_result, load_results, to_row, write_xlsx
from experiment_plan import HYPOTHESES, search_configs, topic_configs

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

def find_repo():
    if os.environ.get("LAB_REPO_ROOT"): return Path(os.environ["LAB_REPO_ROOT"]).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent/"scripts/split_data.py").exists() and (parent/"data").exists(): return parent
    raise FileNotFoundError("Set LAB_REPO_ROOT to the repository containing data/ and scripts/")

class Lab:
    def __init__(self, out=None, device=None, force=False):
        self.repo=find_repo()
        self.out=Path(out).resolve() if out else Path(__file__).resolve().parent.parent
        for name in ("figures","results"): (self.out/name).mkdir(parents=True,exist_ok=True)
        source_dir=Path(__file__).resolve().parent
        (self.out/"code").mkdir(exist_ok=True)
        if source_dir != self.out/"code":
            for source in source_dir.iterdir():
                if source.is_file() and source.suffix in (".py",".ipynb",".txt",".json",".md"):
                    shutil.copy2(source,self.out/"code"/source.name)
        self.checkpoints=self.repo/".work"/self.out.name
        self.checkpoints.mkdir(parents=True,exist_ok=True)
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(1)
        self.force=force
        # Hash the supplied split to make accidental changes detectable on resume.
        split_hash=hashlib.sha256((self.repo/"data/split_metadata.csv").read_bytes()).hexdigest()
        previous=self.out/"run_manifest.json"
        if previous.exists() and not force:
            self.manifest=json.loads(previous.read_text(encoding="utf-8"))
            if self.manifest["split_sha256"]!=split_hash: raise ValueError("Split metadata changed")
        else:
            self.manifest=dict(split_sha256=split_hash,split_seed=42,val_fraction=0.2,
                hypotheses=HYPOTHESES,environment=dict(python=sys.version,torch=torch.__version__,
                platform=platform.platform(),device=self.device,
                gpu=torch.cuda.get_device_name(0) if self.device=="cuda" else None),
                bf16_native=torch.cuda.is_bf16_supported(including_emulation=False) if self.device=="cuda" else False)
        self.save_manifest()
        if not (self.repo/"data/processed/train.npz").exists():
            subprocess.run([sys.executable,"scripts/split_data.py"],cwd=self.repo,check=True,
                           env={**os.environ,"PYTHONIOENCODING":"utf-8"})
        self.data=prepare_data(self.device,processed_dir=self.repo/"data/processed")
        x=self.data["X_tr"][:,:10]
        assert float(x.mean(0).abs().max()) < 1e-4
        assert float((x.std(0,correction=0)-1).abs().max()) < 1e-4
        self.manifest["majority_val_acc"]=self.data["majority_val_acc"]
        self.save_manifest()

    def save_manifest(self): write_json(self.out/"run_manifest.json",self.manifest)

    def health(self):
        path=self.out/"results/health.json"
        if path.exists() and not self.force:
            result=json.loads(path.read_text(encoding="utf-8"))
            print(f"Reuse health: {result['parameters']} parameters, step0={result['step0_loss']:.6f}, overfit20={result['overfit_loss']:.8f}",flush=True)
            return result
        set_seed(1); model=MLP().to(self.device)
        assert count_params(model)==EXPECTED_PARAMS[(256,128)]
        shapes=[]; x=torch.randn(8,54,device=self.device)
        for layer in model.net:
            x=layer(x); shapes.append([type(layer).__name__,list(x.shape)])
        assert tuple(x.shape)==(8,7)
        step0=evaluate(model,self.data["X_val"],self.data["y_val"])["loss"]
        compute_loss(model(self.data["X_tr"][:512]),self.data["y_tr"][:512],"ce").backward()
        grads={name:float(p.grad.norm()) for name,p in model.named_parameters()}
        assert all(value>0 for value in grads.values())
        # A deliberately balanced tiny batch includes every label, with 20 total rows.
        ids=torch.cat([torch.where(self.data["y_tr"]==c)[0][:3 if c<6 else 2] for c in range(7)])
        xb,yb=self.data["X_tr"][ids],self.data["y_tr"][ids]
        set_seed(1); tiny=MLP().to(self.device); opt=torch.optim.Adam(tiny.parameters(),lr=0.01)
        curve=[]
        for step in range(600):
            opt.zero_grad(set_to_none=True); loss=compute_loss(tiny(xb),yb,"ce")
            loss.backward(); opt.step(); curve.append(float(loss.detach()))
        tiny_scores=evaluate(tiny,xb,yb)
        assert tiny_scores["acc"]==1.0 and tiny_scores["loss"]<0.02
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(curve); ax.set(xlabel="Update",ylabel="CE loss",title="Health check: overfit 20 samples")
        ax.set_yscale("log"); fig.tight_layout(); fig.savefig(self.out/"figures/health-overfit20.png",dpi=140); plt.close(fig)
        stats={}
        for init in ("he","xavier","normal","zeros"):
            set_seed(1); m=MLP(init=init).to(self.device)
            stats[init]=dict(activation_std=activation_stats(m,self.data["X_val"][:2048]),
                step0_ce=evaluate(m,self.data["X_val"],self.data["y_val"])["loss"])
        result=dict(parameters=count_params(model),shapes=shapes,step0_loss=step0,
            grad_norms=grads,overfit_loss=tiny_scores["loss"],overfit_acc=tiny_scores["acc"],
            overfit_steps=600,overfit_curve=curve,initialization=stats)
        write_json(path,result)
        print(f"Health OK: {count_params(model)} params, CE step0={step0:.6f}, tiny loss={tiny_scores['loss']:.8f}",flush=True)
        return result

    def run(self,cfg,need_state=False):
        path=self.out/"results"/f"{cfg['exp_id']}.json"
        state_path=self.checkpoints/f"{cfg['exp_id']}.pt"
        if path.exists() and not self.force:
            result=json.loads(path.read_text(encoding="utf-8"))
            if result["cfg"]!=json.loads(json.dumps({**DEFAULT_CFG,**cfg})):
                raise ValueError(f"Cached configuration differs: {cfg['exp_id']}")
            if not need_state or state_path.exists():
                if need_state: result["best_state"]=torch.load(state_path,map_location="cpu",weights_only=True)
                print(f"Reuse measured run {cfg['exp_id']}",flush=True)
                return result
        if self.manifest.get("selection_fixed_before_eval") and not need_state:
            raise ValueError("Study already evaluated. Use a fresh output directory for new experiments.")
        print(f"HYPOTHESIS [{cfg['exp_id']}]: {cfg.get('hypothesis',HYPOTHESES[cfg['group']])}",flush=True)
        # Record the hypothesis on disk before the first optimization step.
        self.manifest.setdefault("planned",{})[cfg["exp_id"]]=cfg
        self.save_manifest()
        result=run_experiment(cfg,self.data)
        save_result(result,self.out/"results")
        plot_run(result,self.out/"figures"/f"{cfg['exp_id']}.png")
        torch.save(result["best_state"],state_path)
        self.refresh_table()
        return result

    def search(self):
        results=[self.run(c) for c in search_configs(DEFAULT_CFG)]
        viable=[r for r in results if not r["summary"]["diverged"]]
        baseline=max((r for r in viable if r["cfg"]["optimizer"]=="sgd_momentum"),key=lambda r:r["summary"]["val_macro_f1"])
        self.manifest["baseline_lr"]=baseline["cfg"]["lr"]
        self.manifest["lr_search_winners"]={opt:max((r for r in viable if r["cfg"]["optimizer"]==opt),
            key=lambda r:r["summary"]["val_macro_f1"])["cfg"]["exp_id"] for opt in ("sgd_momentum","adam")}
        self.save_manifest()
        return results

    def baseline_cfg(self,seed=1):
        return {**DEFAULT_CFG,"lr":self.manifest["baseline_lr"],"seed":seed,"exp_id":f"base-s{seed}",
                "hypothesis":HYPOTHESES["baseline"],"notes":"LR selected on validation; identical split seed 42."}

    def topics(self):
        if "baseline_lr" not in self.manifest: self.search()
        baselines=[self.run(self.baseline_cfg(s)) for s in (1,2,3)]
        # Half the observed baseline median epoch mean gives frequent clipping, not a no-op.
        clip=float(np.median(baselines[0]["history"]["grad_norm"])*0.5)
        self.manifest["clip_norm"]=clip
        self.manifest["clip_rule"]="0.5 * median baseline epoch mean global gradient norm"
        self.save_manifest()
        results=[self.run(c) for c in topic_configs(self.baseline_cfg(),clip,self.device=="cuda")]
        self.comparisons()
        return baselines+results

    def comparisons(self):
        results=load_results(self.out/"results")
        base=next((r for r in results if r["cfg"]["exp_id"]=="base-s1"),None)
        for group in ("optimizer","loss","hparam","dropout","clipping","amp","init","final"):
            group_runs=[r for r in results if r["cfg"]["group"]==group]
            if group_runs:
                plot_compare(([base] if base else [])+group_runs,"val_macro_f1",
                             self.out/"figures"/f"compare_{group}.png",f"{group}: validation macro-F1")

    def refresh_table(self):
        results=load_results(self.out/"results"); rows=[]
        selected=self.manifest.get("selected_exp_id")
        for r in results:
            scores=None
            score_file=self.manifest.get("eval_files",{}).get(r["cfg"]["exp_id"])
            if score_file and (self.out/score_file).exists():
                scores=json.loads((self.out/score_file).read_text())
            if r["cfg"]["exp_id"]=="base-s1" and (self.out/"baseline_eval_result.json").exists():
                scores=json.loads((self.out/"baseline_eval_result.json").read_text())
            if r["cfg"]["exp_id"]==selected and (self.out/"eval_result.json").exists():
                scores=json.loads((self.out/"eval_result.json").read_text())
            notes=""
            if r["cfg"]["precision"]=="fp32" and r["cfg"]["exp_id"]=="base-s1" and not self.manifest["bf16_native"]:
                notes="BF16 skipped: no native hardware support; AMP comparison uses FP16."
            if r["cfg"]["exp_id"]=="base-s1":
                health_path=self.out/"results/health.json"
                if health_path.exists():
                    health=json.loads(health_path.read_text(encoding="utf-8"))
                    notes += f" Majority val acc={self.manifest['majority_val_acc']:.9f}; health step0={health['step0_loss']:.9f}; overfit20 loss={health['overfit_loss']:.12f}, acc={health['overfit_acc']}, updates={health['overfit_steps']}; health/details in results/health.json."
                notes += " Diagnostics and initialization activation_std are stored in results JSON; peak includes resident data; time includes epoch evaluation."
            rows.append(to_row(r,scores,notes))
        write_xlsx(rows,self.repo/"templates/experiment_table_template.xlsx",self.out/"experiments.xlsx")

    def finalize(self):
        results=load_results(self.out/"results")
        if "selected_exp_id" not in self.manifest:
            # Fix selection BEFORE calling the official evaluation script.
            valid=[r for r in results if not r["summary"]["diverged"] and r["cfg"]["seed"]==1
                   and r["summary"]["completed_epochs"]==r["cfg"]["epochs"]]
            chosen=max(valid,key=lambda r:r["summary"]["val_macro_f1"])
            self.manifest.update(selected_exp_id=chosen["cfg"]["exp_id"],selected_cfg=chosen["cfg"],
                selection_rule="Highest validation macro-F1 of best-val-loss checkpoints, seed 1, completed runs",
                selection_fixed_before_eval=True)
            self.save_manifest()
        chosen=next(r for r in results if r["cfg"]["exp_id"]==self.manifest["selected_exp_id"])
        baseline=next(r for r in results if r["cfg"]["exp_id"]=="base-s1")
        # Config and submission seed stay fixed; replicas measure variability only.
        replicas=[]
        for seed in (2,3):
            replica_cfg={**chosen["cfg"],"seed":seed,"exp_id":f"final-s{seed}","group":"final",
                "description":"Replica of validation-selected final configuration",
                "hypothesis":HYPOTHESES["final"],
                "notes":f"Same selected config as {chosen['cfg']['exp_id']}; seed changed only. Not selected using eval."}
            replicas.append(self.run(replica_cfg,need_state=True))
        self.manifest["final_replica_ids"]=[r["cfg"]["exp_id"] for r in replicas]
        self.manifest["eval_files"]={"base-s1":"baseline_eval_result.json",
            "base-s2":"eval_base_s2.json","base-s3":"eval_base_s3.json",
            chosen["cfg"]["exp_id"]:"eval_result.json","final-s2":"eval_final_s2.json","final-s3":"eval_final_s3.json"}
        self.save_manifest()
        evaluations=[(baseline,self.out/"predictions_baseline.csv","baseline_eval_result.json"),
            (chosen,self.out/"predictions_eval.csv","eval_result.json")]
        for seed in (2,3):
            baseline_replica=next(r for r in results if r["cfg"]["exp_id"]==f"base-s{seed}")
            evaluations.append((baseline_replica,self.checkpoints/f"predictions_base_s{seed}.csv",f"eval_base_s{seed}.json"))
            evaluations.append((replicas[seed-2],self.checkpoints/f"predictions_final_s{seed}.csv",f"eval_final_s{seed}.json"))
        for r,pred_path,score_name in evaluations:
            if (self.out/score_name).exists() and not self.force:
                print(f"Reuse fixed official score {score_name}; selection remains frozen",flush=True)
                continue
            model_result=self.run(r["cfg"],need_state=True)
            final_eval(r["cfg"],model_result,self.data,pred_path)
            completed=subprocess.run([sys.executable,str(self.repo/"scripts/evaluate.py"),
                "--pred",str(pred_path),"--out",str(self.out/score_name)],cwd=self.repo,
                text=True,encoding="utf-8",capture_output=True,check=True,
                env={**os.environ,"PYTHONIOENCODING":"utf-8"})
            print(completed.stdout,flush=True)
        self.refresh_table(); self.comparisons()
        from report import write_report
        write_report(self.out,self.manifest)
        return json.loads((self.out/"eval_result.json").read_text())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="submission_MSSV")
    parser.add_argument("--stage",choices=("health","search","topics","final","all"),default="all")
    parser.add_argument("--device",choices=("cpu","cuda"))
    parser.add_argument("--force",action="store_true",help="Start new training; remove old eval outputs first")
    args=parser.parse_args()
    lab=Lab(args.out,args.device,args.force)
    if args.force and (lab.out/"eval_result.json").exists():
        raise ValueError("Use a fresh output directory for a new study; do not tune after eval")
    lab.health()
    if args.stage in ("search","all"): lab.search()
    if args.stage in ("topics","all"): lab.topics()
    if args.stage in ("final","all"): lab.finalize()

if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    main()
