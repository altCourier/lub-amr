"""
tests/test_amr_dataset_real_data.py

Integration tests that run AMRDataset against the REAL split files
(data/splits/train.h5, val.h5, test.h5), as opposed to test_amr_dataset.py
which uses a small synthetic fixture.

These verify the actual data on disk conforms to what AMRDataset expects --
e.g. no unexpected NaNs, expected split sizes/ratios, correct value ranges --
catching data issues that a synthetic fixture can't.

If the real files aren't present (e.g. running in CI without the dataset,
or from a machine that hasn't generated data/splits/ yet), these tests are
SKIPPED rather than failed, so the rest of the suite still runs cleanly.

Run with:
    pytest tests/ -v
    pytest tests/test_amr_dataset_real_data.py -v   # just this file
"""

import numpy as np
import pytest
import torch
import h5py

from torch.utils.data import DataLoader

from amr_dataset import AMRDataset, compute_global_stats, MOD_NAMES
from paths import TRAIN_PATH, VAL_PATH, TEST_PATH

N_MODS = 5
SEQ_LEN = 1024
EXPECTED_SNRS = {-10.0, -8.0, -6.0, -4.0, -2.0, 0.0,
                 2.0, 4.0, 6.0, 8.0, 10.0,
                 12.0, 14.0, 16.0, 18.0, 20.0}

split_files = [TRAIN_PATH, VAL_PATH, TEST_PATH]
any_missing = any(not p.exists() for p in split_files)

pytestmark = pytest.mark.skipif(
    any_missing,
    reason=(
        "Real split files not found under data/splits/. "
        "Skipping real-data integration tests -- run from a machine with "
        "the dataset generated, or update SPLITS_DIR in this file."
    ),
)


# ---------------------------------------------------------------------------
# Basic structural checks on each real split file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_name,path", [
    ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
])
def test_real_split_has_expected_keys(split_name, path):
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
    assert {"Data", "Mods", "SNRs", "Domain"}.issubset(keys), (
        f"{split_name}.h5 missing expected keys, found: {keys}"
    )


@pytest.mark.parametrize("split_name,path", [
    ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
])
def test_real_split_shapes_consistent(split_name, path):

    with h5py.File(path, "r") as f:

        n = f["Data"].shape[0]

        assert f["Data"].shape == (n, SEQ_LEN)
        assert f["Mods"].shape == (n, N_MODS)
        assert f["SNRs"].shape == (n,)
        assert f["Domain"].shape == (n,)


# ---------------------------------------------------------------------------
# Split ratio sanity (should be ~80/10/10, per-group stratified)
# ---------------------------------------------------------------------------

def test_real_split_ratios_roughly_80_10_10():

    with h5py.File(TRAIN_PATH, "r") as f:
        n_train = f["Data"].shape[0]

    with h5py.File(VAL_PATH, "r") as f:
        n_val = f["Data"].shape[0]

    with h5py.File(TEST_PATH, "r") as f:
        n_test = f["Data"].shape[0]

    total = n_train + n_val + n_test

    assert n_train / total == pytest.approx(0.8, abs=0.01)
    assert n_val / total == pytest.approx(0.1, abs=0.01)
    assert n_test / total == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("split_name,path", [
    ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
])
def test_real_split_domain_balance(split_name, path):
    """
    Each split should be ~50/50 Rma/Umi, since the pooled dataset is balanced.
    """

    with h5py.File(path, "r") as f:
        domain = f["Domain"][:]

    umi_frac = domain.mean()

    assert umi_frac == pytest.approx(0.5, abs=0.02), (
        f"{split_name}.h5 domain imbalance: {umi_frac:.3f} fraction Umi"
    )


@pytest.mark.parametrize("split_name,path", [
    ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
])
def test_real_split_mod_balance(split_name, path):
    """
    Each split should have roughly equal counts per modulation class.
    """

    with h5py.File(path, "r") as f:
        mods = f["Mods"][:]

    class_counts = mods.sum(axis=0)
    mean_count = class_counts.mean()

    for i, count in enumerate(class_counts):

        assert count == pytest.approx(mean_count, rel=0.05), (
            f"{split_name}.h5 class {MOD_NAMES[i]} count {count} deviates >5% from mean {mean_count:.1f}"
        )


@pytest.mark.parametrize("split_name,path", [
    ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
])
def test_real_split_mods_are_onehot(split_name, path):

    with h5py.File(path, "r") as f:
        mods = f["Mods"][:]

    row_sums = mods.sum(axis=1)

    assert (row_sums == 1).all(), f"{split_name}.h5 has non-one-hot Mods rows"


@pytest.mark.parametrize("split_name,path", [
    ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
])
def test_real_split_no_1024qam_leakage(split_name, path):
    """
    Confirms Mods has exactly 5 columns
    """

    with h5py.File(path, "r") as f:
        n_classes = f["Mods"].shape[1]

    assert n_classes == 5, f"{split_name}.h5 has {n_classes} mod classes, expected 5"


# ---------------------------------------------------------------------------
# No leakage between splits (no duplicate rows across train/val/test)
# ---------------------------------------------------------------------------

def test_no_duplicate_rows_across_splits():
    """
    Spot-checks that train/val/test don't share identical Data rows -- i.e. the
    stratified split didn't accidentally place the same sample in two splits.
    Uses a hash of each row's bytes for a fast approximate check rather than
    an O(n^2) full comparison.
    """

    def row_hashes(path, max_rows=None):

        with h5py.File(path, "r") as f:
            data = f["Data"][:max_rows]

        return set(hash(row.tobytes()) for row in data)

    train_hashes = row_hashes(TRAIN_PATH)
    val_hashes = row_hashes(VAL_PATH)
    test_hashes = row_hashes(TEST_PATH)

    assert train_hashes.isdisjoint(val_hashes), "train/val share identical rows"
    assert train_hashes.isdisjoint(test_hashes), "train/test share identical rows"
    assert val_hashes.isdisjoint(test_hashes), "val/test share identical rows"


# ---------------------------------------------------------------------------
# AMRDataset against the real files: no NaNs, correct shapes, DataLoader works
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [TRAIN_PATH, VAL_PATH, TEST_PATH])
@pytest.mark.parametrize("normalize", ["per_sample", None])
def test_real_data_no_nans_sampled(path, normalize):
    """
    Full-file NaN check would be slow on ~1GB+ files, so this samples a
    random subset of indices per split rather than iterating every row.
    """
    ds = AMRDataset(str(path), normalize=normalize, preload=False)
    rng = np.random.default_rng(0)
    n_check = min(200, len(ds))
    sample_idx = rng.choice(len(ds), size=n_check, replace=False)

    for i in sample_idx:
        iq, mod_label, snr, domain = ds[int(i)]
        assert torch.isfinite(iq).all(), f"non-finite values at real index {i} in {path.name}"
        assert 0 <= mod_label.item() < N_MODS
        assert domain.item() in (0, 1)

    ds.close()


def test_real_train_works_with_dataloader():

    ds = AMRDataset(str(TRAIN_PATH), normalize="per_sample", preload=False)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    iq_batch, mod_batch, snr_batch, domain_batch = next(iter(loader))

    assert iq_batch.shape == (64, 2, SEQ_LEN)
    assert mod_batch.shape == (64,)
    assert snr_batch.shape == (64,)
    assert domain_batch.shape == (64,)

    ds.close()


def test_real_global_stats_computable_on_train():
    """
    compute_global_stats should run cleanly on the real train.h5 and return
    sane (mean, std) values. This is also a reminder: compute on train ONLY,
    reuse the same numbers for val/test to avoid leakage.
    """
    mean, std = compute_global_stats(str(TRAIN_PATH))
    assert np.isfinite(mean)
    assert np.isfinite(std)
    assert std > 0