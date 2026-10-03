"""Fixed split, train-only standardization and device-resident mini-batches."""
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10

def load_split(processed_dir="data/processed"):
    with np.load(Path(processed_dir) / "train.npz") as tr, np.load(Path(processed_dir) / "eval.npz") as ev:
        arrays = (tr["X"], tr["y"], ev["X"], ev["y"], ev["row_id"])
    for X, y in ((arrays[0], arrays[1]), (arrays[2], arrays[3])):
        assert X.ndim == 2 and X.shape[1] == 54 and X.dtype == np.float32
        assert y.shape == (len(X),) and y.dtype == np.int64
        assert np.isfinite(X).all() and y.min() >= 0 and y.max() < 7
    return arrays

def make_val_split(X, y, val_fraction=0.2, seed=42):
    return train_test_split(X, y, test_size=val_fraction, stratify=y, random_state=seed)

def fit_standardizer(X_tr):
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean, std = numeric.mean(0), numeric.std(0)
    return mean, np.where(std > 0, std, 1.0)

def apply_standardizer(X, mean, std):
    out = X.copy()
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / std
    return out

def prepare_data(device, val_fraction=0.2, seed=42, processed_dir="data/processed"):
    X, y, Xe, ye, ids = load_split(processed_dir)
    Xt, Xv, yt, yv = make_val_split(X, y, val_fraction, seed)
    mean, std = fit_standardizer(Xt)
    data = {"eval_row_id": ids, "mean": mean, "std": std,
            "split_seed": seed, "val_fraction": val_fraction}
    for name, features, labels in (("tr", Xt, yt), ("val", Xv, yv), ("eval", Xe, ye)):
        data[f"X_{name}"] = torch.as_tensor(apply_standardizer(features, mean, std), device=device)
        data[f"y_{name}"] = torch.as_tensor(labels, device=device)
        print(f"{name}: {tuple(data[f'X_{name}'].shape)}, counts {np.bincount(labels, minlength=7).tolist()}")
    data["majority_val_acc"] = float(np.mean(yv == np.bincount(yt).argmax()))
    print(f"Majority validation accuracy: {data['majority_val_acc']:.6f}")
    return data

def iterate_batches(X, y, batch_size, generator=None, shuffle=True):
    if batch_size <= 0: raise ValueError("batch_size must be positive")
    if shuffle:
        perm = torch.randperm(len(X), generator=generator, device=X.device)
        for start in range(0, len(X), batch_size):
            idx = perm[start:start + batch_size]
            yield X[idx], y[idx]
    else:
        for start in range(0, len(X), batch_size):
            yield X[start:start + batch_size], y[start:start + batch_size]
    # Keep the last partial batch; every row is used once per epoch.
