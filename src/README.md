# viz.py

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

# train.py

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

# paths.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR             = PROJECT_ROOT / 'data'
DATA_RAW_DIR         = DATA_DIR / 'raw'
DATA_PROCESSED_DIR   = DATA_DIR / 'processed'
DATA_SPLITS_DIR      = DATA_DIR / 'splits'

TRAIN_PATH           = DATA_SPLITS_DIR / 'train.h5'
VAL_PATH             = DATA_SPLITS_DIR / 'val.h5'
TEST_PATH            = DATA_SPLITS_DIR / 'test.h5'

RMA_H5_DIR           = DATA_RAW_DIR / 'Rma.h5'
UMI_H5_DIR           = DATA_RAW_DIR / 'Umi.h5'
RMA_CLEAN_DIR        = DATA_PROCESSED_DIR / 'Rma_clean.h5'
UMI_CLEAN_DIR        = DATA_PROCESSED_DIR / 'Umi_clean.h5'

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# models.py

Model architectures for AMR.

Every model here takes input of shape: 
(batch, 2, 1024) -- the (I, Q) tensor produced by AMRDataset (see amr_dataset.py) 
                 -- and outputs raw logits over
                 
the 5 modulation classes (see MOD_NAMES in amr_dataset.py).

# amr_dataset.py

Reusable PyTorch Dataset for the AMR project.

    Reads samples from the split .h5 files produced by the cleaning/splitting notebook
    (data/splits/train.h5, val.h5, test.h5), each containing:
        Data:   complex64, shape (N, 1024)  -- raw I/Q symbol vectors
        Mods:   float32,   shape (N, 5)     -- one-hot modulation label
        SNRs:   float32,   shape (N,)       -- Eb/N0 value (dB)
        Domain: int8,      shape (N,)       -- 0 = Rma, 1 = Umi

Converts each complex64 row into a real-valued 2-channel (I, Q) tensor of shape
(2, 1024), the standard input format for 1D-CNN / CNN-LSTM AMR models.

Usage:
    from amr_dataset import AMRDataset
    from torch.utils.data import DataLoader

    train_ds = AMRDataset("data/splits/train.h5", normalize="per_sample")
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    for iq, mod_label, snr, domain in train_loader:
        ...

# utils.py

Small shared helpers used across train.py, notebooks, and eval scripts.
def get_device() -> torch.device:
        
