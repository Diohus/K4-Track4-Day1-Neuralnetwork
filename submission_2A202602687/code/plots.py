"""PNG evidence for each run, including failures, and group comparisons."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_run(result, path):
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = h["epoch"]
    axes[0].plot(epochs, h["train_loss"], label="train (eval mode)")
    axes[0].plot(epochs, h["val_loss"], label="validation")
    axes[0].set_ylabel(f"{cfg['loss'].upper()} loss")
    axes[1].plot(epochs, h["val_acc"], label="accuracy")
    axes[1].plot(epochs, h["val_macro_f1"], label="macro-F1")
    axes[1].set_ylabel("Validation score")
    axes[2].plot(epochs, h["grad_norm"], label="mean L2 before clip")
    if cfg.get("clip_norm") is not None:
        axes[2].axhline(cfg["clip_norm"], ls="--", color="red", label="clip threshold")
    axes[2].set_ylabel("Gradient norm")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        if s.get("best_epoch", 0): ax.axvline(s["best_epoch"], color="gray", ls=":", alpha=0.7)
        if not epochs: ax.text(0.5, 0.5, "No completed epoch / unsupported", ha="center", transform=ax.transAxes)
    fig.suptitle(f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} batch={cfg['batch']} "
                 f"hidden={cfg['hidden']} p={cfg['dropout']} init={cfg['init']} {cfg['precision']} "
                 f"seed={cfg['seed']}" + (" | DIVERGED" if s.get("diverged") else ""), fontsize=10)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)

def plot_compare(results, metric, path, title=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        ax.plot(r["history"]["epoch"], r["history"][metric], label=r["cfg"]["exp_id"])
    ax.set(xlabel="Epoch", ylabel=metric, title=title or metric)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
