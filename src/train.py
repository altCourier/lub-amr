"""
train.py

Training loop for AMR models. 

Thin orchestrator: 

    --      imports AMRDataset (data)
    --      and a model class from models.py (architecture), and handles the actual
            epoch loop, optimizer, loss, validation, logging, and checkpointing.

            

Usage (from project root, or adjust sys.path as needed):

    python src/train.py

Or import `train_model` directly from a notebook for interactive experiments:

    from train import train_model
    from models import BaselineCNN
    history = train_model(model_cls=BaselineCNN, run_name="baseline_pooled", epochs=20)

"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from amr_dataset import AMRDataset
from models import BaselineCNN
from paths import TRAIN_PATH, VAL_PATH, PROJECT_ROOT, CHECKPOINT_DIR
from utils import get_device

def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
        Runs a single pass over `loader`.
    
        If train=True, updates model weights;
        If train=False (validation), runs in eval/no_grad mode and does not
            update weights. 
    
        Returns (avg_loss, accuracy) for the epoch.
    """

    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:

        for iq, mod_label, snr, domain in loader:
            iq = iq.to(device, non_blocking=True)
            mod_label = mod_label.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(iq)
            loss = criterion(logits, mod_label)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = iq.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == mod_label).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_model(
    model_cls        = BaselineCNN,
    run_name: str    = "baseline_pooled",
    epochs: int      = 20,
    batch_size: int  = 256,
    lr: float        = 1e-3,
    normalize: str   = "per_sample",
    train_path       = TRAIN_PATH,
    val_path         = VAL_PATH,
    domain_filter    = None,
    num_workers: int = 4,
    patience: int    = 5,
):
    """
    Trains `model_cls` on `train_path`, validates on `val_path` each epoch,
    saves the best checkpoint and returns the per-epoch history.

    Parameters
    ----------
        model_cls : nn.Module subclass (not instance) from models.py

        run_name : str
            Used to name the saved checkpoint file for example "baseline_pooled",
            "baseline_rma_only", "baseline_umi_only".

        domain_filter : int or None
            Passed through to AMRDataset if/when domain filtering is added there
            (0=Rma, 1=Umi, None=pooled). Currently a placeholder for the
            Rma-only / Umi-only baseline runs (TBD)

        patience : int
            Stop early if validation accuracy doesn't improve for this many
            consecutive epochs (0 disables early stopping).

    Returns
    -------
    history : dict with keys "train_loss", "train_acc", "val_loss", "val_acc",
        each a list of per-epoch values.
    """
    
    device = get_device()
    print(f"Using device: {device}")

    if domain_filter is not None:
        # domain_filter : int or None
        #     0 = Rma, 1 = Umi, None = pooled. 
        #     Applied to both train_ds and val_ds so
        #     early stopping/checkpointing reflect the same domain being trained
        #     on. test.h5 stays unfiltered/pooled at eval time for every variant,
        #     so all baselines are compared on the same fixed test set.

        train_ds = AMRDataset(str(train_path), normalize = normalize, domain_filter = domain_filter)
        val_ds   = AMRDataset(str(val_path), normalize = normalize, domain_filter = domain_filter)

    train_ds = AMRDataset(str(train_path), normalize = normalize)
    val_ds   = AMRDataset(str(val_path), normalize = normalize)

    train_loader = DataLoader(
        train_ds, batch_size = batch_size, shuffle = True,
        num_workers = num_workers, pin_memory = (device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds, batch_size = batch_size, shuffle = False,
        num_workers = num_workers, pin_memory = (device.type == "cuda"),
    )

    model     = model_cls().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}_best.pt"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"[{run_name}] epoch {epoch:3d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "run_name": run_name,
                },
                checkpoint_path,
            )
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"  -> no improvement for {patience} epochs, stopping early")
                break

    return history


if __name__ == "__main__":
    train_model()
