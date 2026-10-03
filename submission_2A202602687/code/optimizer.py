"""Optimizer factory and global L2 norm measured before clipping."""
import torch
OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")

def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9, betas=(0.9, 0.999), eps=1e-8):
    if lr is None or lr <= 0: raise ValueError("Choose a positive learning rate using validation")
    common = dict(lr=lr, weight_decay=weight_decay)
    if name == "sgd": return torch.optim.SGD(params, **common)
    if name == "sgd_momentum": return torch.optim.SGD(params, momentum=momentum, **common)
    if name == "adam": return torch.optim.Adam(params, betas=betas, eps=eps, **common)
    if name == "adamw": return torch.optim.AdamW(params, betas=betas, eps=eps, **common)
    raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(optimizer, name, total_steps, **kwargs):
    if name is None: return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")

def clip_gradients(params, max_norm):
    return float(torch.nn.utils.clip_grad_norm_(params, float("inf") if max_norm is None else max_norm))
