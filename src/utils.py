"""
utils.py

Small shared helpers used across train.py, notebooks, and eval scripts.
"""

import torch
import numpy as np
from models import BaselineCNN

from paths import CHECKPOINT_DIR

MOD_NAMES = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM']

def get_device() -> torch.device:
    """Returns cuda if available, else cpu."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def load_best_model(run_name, model_cls=BaselineCNN, device=torch.device, CHECKPOINT_DIR = CHECKPOINT_DIR):

    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    m = model_cls().to(device)
    m.load_state_dict(ckpt["model_state_dict"])

    m.eval()

    print(f"Loaded {run_name}: epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    return m

def high_snr_confusion(mod_true, mod_pred, snr, threshold=15.0, mod_names=MOD_NAMES):

    mask = snr >= threshold

    n = len(mod_names)
    cm = np.zeros((n, n), dtype=np.float64)

    for t, p in zip(mod_true[mask], mod_pred[mask]):
        cm[t, p] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    cm_norm = cm / row_sums

    print(f"n={mask.sum()} samples at SNR>={threshold}")
    print(f"{'':>8}" + "".join(f"{m:>8}" for m in mod_names))

    for i, name in enumerate(mod_names):
        print(f"{name:>8}" + "".join(f"{cm_norm[i,j]:8.3f}" for j in range(n)))

    return cm_norm
