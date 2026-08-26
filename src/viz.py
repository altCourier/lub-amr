"""
viz.py

Reusable evaluation + visualization functions for AMR experiments.

Since every experiment (baseline, curriculum variants, Rma-only/Umi-only,
etc.) needs the SAME kind of results -- training curves, accuracy broken
down by SNR, confusion matrices -- these live here once rather than being
rewritten per notebook.

Functions:
    plot_training_curves(history)
    collect_predictions(model, loader, device)
    plot_accuracy_vs_snr(mod_true, mod_pred, snr, ...)
    plot_confusion_matrix(mod_true, mod_pred, ...)
    accuracy_vs_snr_table(mod_true, mod_pred, snr, ...)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

MOD_NAMES = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM']

def plot_training_curves(history, title=None, ax=None):
    """
    Plots train/val loss and accuracy side by side from the `history` dict
    returned by train.train_model().
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="val")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    return ax


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Runs `model` over every batch in `loader` (e.g. the test set) and
    collects predictions alongside ground truth, SNR, and domain -- the raw
    material for accuracy-vs-SNR curves and confusion matrices.

    Returns
    -------
    mod_true, mod_pred : np.ndarray, shape (N,) -- class indices
    snr : np.ndarray, shape (N,)
    domain : np.ndarray, shape (N,)
    """
    model.eval()

    all_true, all_pred, all_snr, all_domain = [], [], [], []

    for iq, mod_label, snr, domain in loader:
        iq = iq.to(device, non_blocking=True)
        logits = model(iq)
        pred = logits.argmax(dim=1).cpu().numpy()

        all_true.append(mod_label.numpy())
        all_pred.append(pred)
        all_snr.append(snr.numpy())
        all_domain.append(domain.numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_snr),
        np.concatenate(all_domain),
    )


def accuracy_vs_snr_table(mod_true, mod_pred, snr, mod_names=MOD_NAMES):
    """
    Returns a dict: {mod_name: {snr_level: accuracy}}, plus an "overall" key
    for accuracy across all modulations at each SNR level. Useful for both
    plotting and for a plain numeric readout / comparison across experiments.
    """
    snr_levels = np.unique(snr)
    correct = (mod_true == mod_pred)

    table = {"overall": {}}
    for s in snr_levels:
        mask = snr == s
        table["overall"][s] = correct[mask].mean() if mask.sum() > 0 else np.nan

    for i, name in enumerate(mod_names):
        table[name] = {}
        for s in snr_levels:
            mask = (snr == s) & (mod_true == i)
            table[name][s] = correct[mask].mean() if mask.sum() > 0 else np.nan

    return table


def plot_accuracy_vs_snr(mod_true, mod_pred, snr, mod_names=MOD_NAMES, title=None, ax=None):
    """
    Standard AMR evaluation plot: accuracy (y) vs SNR (x), one line per
    modulation class plus an overall line. This is the primary comparison
    tool across experiments (baseline vs curriculum variants, etc.).
    """
    table = accuracy_vs_snr_table(mod_true, mod_pred, snr, mod_names)
    snr_levels = sorted(table["overall"].keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for name in mod_names:
        accs = [table[name][s] for s in snr_levels]
        ax.plot(snr_levels, accs, marker="o", label=name)

    overall_accs = [table["overall"][s] for s in snr_levels]
    ax.plot(snr_levels, overall_accs, marker="s", linestyle="--",
             color="black", label="overall", linewidth=2)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "Accuracy vs SNR")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_confusion_matrix(mod_true, mod_pred, mod_names=MOD_NAMES, normalize=True, title=None, ax=None):
    """
    Confusion matrix over the full test set (all SNRs pooled). Useful for
    seeing which modulation pairs get confused most often (e.g. 64QAM vs
    256QAM at low SNR is a common AMR failure mode).
    """
    n = len(mod_names)
    cm = np.zeros((n, n), dtype=np.float64)

    for t, p in zip(mod_true, mod_pred):
        cm[t, p] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid div by zero
        cm = cm / row_sums

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(mod_names, rotation=45, ha="right")
    ax.set_yticklabels(mod_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix" + (" (normalized)" if normalize else ""))

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                     color="white" if val > 0.5 else "black", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return ax

def plot_accuracy_vs_snr_comparison(results, mod_names=MOD_NAMES, title=None, ax=None):
    """
    Overlays the *overall* accuracy-vs-SNR curve for multiple experiments on
    one axis -- e.g. pooled vs Rma-only vs Umi-only baselines, or later,
    curriculum variants vs baseline. Unlike plot_accuracy_vs_snr (which shows
    one model's per-class breakdown), this compares overall accuracy across
    models, all evaluated on the same test set.

    Parameters
    ----------
    results : dict[str, tuple]
        Maps a label (e.g. "pooled", "rma_only") to a (mod_true, mod_pred, snr)
        tuple -- the output of collect_predictions() for that model, run on
        the SAME test loader.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for label, (mod_true, mod_pred, snr) in results.items():
        table = accuracy_vs_snr_table(mod_true, mod_pred, snr, mod_names)
        snr_levels = sorted(table["overall"].keys())
        overall_accs = [table["overall"][s] for s in snr_levels]
        ax.plot(snr_levels, overall_accs, marker="o", label=label)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "Accuracy vs SNR — Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
