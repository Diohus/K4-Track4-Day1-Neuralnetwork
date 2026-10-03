"""Numerical checks for loss aggregation, F1, clipping and the required models."""
import uuid
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from data import fit_standardizer, apply_standardizer
from model import MLP, EXPECTED_PARAMS, count_params
from train import compute_loss, evaluate, macro_f1_from_confusion, write_predictions
from optimizer import clip_gradients

def validate_numerics():
    torch.set_num_threads(1); torch.manual_seed(7)
    for hidden, expected in EXPECTED_PARAMS.items():
        m=MLP(hidden)
        assert count_params(m)==expected and m(torch.randn(9,54)).shape==(9,7)
    X=torch.randn(23,54); y=torch.arange(23)%7; m=MLP()
    m.eval()
    for loss in ("ce","mse"):
        measured=evaluate(m,X,y,loss,batch_size=8)
        reference=float(compute_loss(m(X),y,loss).detach())
        assert abs(measured["loss"]-reference)<1e-6, (loss,measured,reference)
        preds=m(X).argmax(1).detach().numpy()
        assert abs(measured["macro_f1"]-f1_score(y.numpy(),preds,labels=range(7),average="macro",zero_division=0))<1e-12
    cm=np.zeros((7,7)); cm[0,0]=3; cm[0,1]=2
    assert np.isfinite(macro_f1_from_confusion(cm))
    p=torch.nn.Parameter(torch.zeros(2)); p.grad=torch.tensor([3.,4.])
    assert abs(clip_gradients([p],2.)-5.)<1e-6
    assert abs(float(p.grad.norm())-2.)<1e-5
    a=np.zeros((5,54),dtype=np.float32); a[:,:10]=np.arange(5)[:,None]
    a[:,10:]=1
    mean,std=fit_standardizer(a); b=apply_standardizer(a,mean,std)
    assert np.max(np.abs(b[:,:10].mean(0)))<1e-6
    assert np.array_equal(b[:,10:],a[:,10:]) and np.array_equal(a[:,0],np.arange(5))
    shifted=apply_standardizer(a+100,mean,std)
    assert shifted[:,0].mean()>10  # Validation must not fit its own statistics.
    # An ordinary unique file avoids restrictive Windows temp-directory ACLs.
    path=Path(__file__).resolve().parent/f".check_predictions_{uuid.uuid4().hex}.csv"
    try:
        write_predictions(np.array([99,42]),np.array([0,6]),path)
        assert path.read_text().splitlines()==["row_id,pred","99,0","42,6"]
        try: write_predictions(np.array([1,1]),np.array([0,6]),path)
        except ValueError: pass
        else: raise AssertionError("Duplicate IDs accepted")
    finally:
        if path.exists(): path.unlink()
    print("Numerical validation passed: models, CE/MSE uneven batches, F1, pre-clip norm, train-only scaling, CSV.")
    return True

if __name__=="__main__": validate_numerics()
