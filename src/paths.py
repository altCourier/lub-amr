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
