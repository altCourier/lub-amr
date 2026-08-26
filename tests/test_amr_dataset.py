"""
tests/test_amr_dataset.py

Sanity tests for AMRDataset (src/amr_dataset.py).

These tests run against a small SYNTHETIC .h5 file built in a pytest fixture,
matching the real schema (Data/Mods/SNRs/Domain) from data/splits/*.h5. This
keeps the test suite fast and independent of the real dataset being present
on disk.

Run with:
    pytest tests/ -v

(requires the project root, or src/, to be importable -- see conftest.py)
"""

import numpy as np
import pytest
import torch
import h5py

from amr_dataset import AMRDataset, compute_global_stats, MOD_NAMES, DOMAIN_NAMES


N_SAMPLES = 40
N_MODS = 5
SEQ_LEN = 1024


@pytest.fixture
def fake_split_file(tmp_path):
    """Builds a small synthetic split file with the same schema as train/val/test.h5."""
    rng = np.random.default_rng(0)

    real = rng.standard_normal((N_SAMPLES, SEQ_LEN)).astype(np.float32)
    imag = rng.standard_normal((N_SAMPLES, SEQ_LEN)).astype(np.float32)
    data = (real + 1j * imag).astype(np.complex64)

    mod_idx = rng.integers(0, N_MODS, size=N_SAMPLES)
    mods = np.eye(N_MODS, dtype=np.float32)[mod_idx]

    snrs = rng.choice([-10.0, 0.0, 10.0, 20.0], size=N_SAMPLES).astype(np.float32)
    domain = rng.integers(0, 2, size=N_SAMPLES).astype(np.int8)

    path = tmp_path / "fake_split.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("Data", data=data)
        f.create_dataset("Mods", data=mods)
        f.create_dataset("SNRs", data=snrs)
        f.create_dataset("Domain", data=domain)

    return str(path)


# ---------------------------------------------------------------------------
# Shape / dtype / length checks
# ---------------------------------------------------------------------------

def test_length_matches_file(fake_split_file):
    ds = AMRDataset(fake_split_file, normalize="per_sample")
    assert len(ds) == N_SAMPLES


def test_item_shapes_and_dtypes(fake_split_file):
    ds = AMRDataset(fake_split_file, normalize="per_sample")
    iq, mod_label, snr, domain = ds[0]

    assert iq.shape == (2, SEQ_LEN)
    assert iq.dtype == torch.float32

    assert mod_label.dtype == torch.long
    assert 0 <= mod_label.item() < N_MODS

    assert snr.dtype == torch.float32
    assert domain.dtype == torch.long
    assert domain.item() in (0, 1)


def test_all_indices_readable(fake_split_file):
    """Every index should be readable without error, not just index 0."""
    ds = AMRDataset(fake_split_file, normalize="per_sample")
    for i in range(len(ds)):
        iq, mod_label, snr, domain = ds[i]
        assert iq.shape == (2, SEQ_LEN)


# ---------------------------------------------------------------------------
# No NaNs / Infs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("normalize", ["per_sample", "global", None])
def test_no_nans_or_infs(fake_split_file, normalize):
    global_stats = None
    if normalize == "global":
        global_stats = compute_global_stats(fake_split_file)

    ds = AMRDataset(fake_split_file, normalize=normalize, global_stats=global_stats)
    for i in range(len(ds)):
        iq, *_ = ds[i]
        assert torch.isfinite(iq).all(), f"non-finite value at index {i} (normalize={normalize})"


# ---------------------------------------------------------------------------
# Normalization behavior
# ---------------------------------------------------------------------------

def test_per_sample_normalization_unit_energy(fake_split_file):
    """After per-sample normalization, mean(I^2 + Q^2) should be ~1.0 per sample."""
    ds = AMRDataset(fake_split_file, normalize="per_sample")
    for i in range(len(ds)):
        iq, *_ = ds[i]
        energy = (iq[0] ** 2 + iq[1] ** 2).mean().item()
        assert energy == pytest.approx(1.0, abs=1e-4)


def test_none_normalization_matches_raw_values(fake_split_file):
    """normalize=None should return the raw I/Q values unchanged."""
    with h5py.File(fake_split_file, "r") as f:
        raw = f["Data"][0]

    ds = AMRDataset(fake_split_file, normalize=None)
    iq, *_ = ds[0]

    np.testing.assert_allclose(iq[0].numpy(), raw.real, atol=1e-6)
    np.testing.assert_allclose(iq[1].numpy(), raw.imag, atol=1e-6)


def test_global_normalization_requires_stats(fake_split_file):
    """normalize='global' without global_stats should raise, not silently misbehave."""
    with pytest.raises(ValueError):
        AMRDataset(fake_split_file, normalize="global", global_stats=None)


def test_compute_global_stats_reasonable(fake_split_file):
    mean, std = compute_global_stats(fake_split_file)
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std > 0


# ---------------------------------------------------------------------------
# Labels / metadata integrity (mod index, SNR, domain match the source file)
# ---------------------------------------------------------------------------

def test_mod_label_matches_onehot_argmax(fake_split_file):
    with h5py.File(fake_split_file, "r") as f:
        mods = f["Mods"][:]

    ds = AMRDataset(fake_split_file, normalize=None)
    for i in range(len(ds)):
        _, mod_label, _, _ = ds[i]
        assert mod_label.item() == np.argmax(mods[i])


def test_snr_and_domain_pass_through_unchanged(fake_split_file):
    with h5py.File(fake_split_file, "r") as f:
        snrs = f["SNRs"][:]
        domain = f["Domain"][:]

    ds = AMRDataset(fake_split_file, normalize=None)
    for i in range(len(ds)):
        _, _, snr, dom = ds[i]
        assert snr.item() == pytest.approx(float(snrs[i]))
        assert dom.item() == int(domain[i])


# ---------------------------------------------------------------------------
# preload=True vs preload=False should return identical data
# ---------------------------------------------------------------------------

def test_preload_true_and_false_agree(fake_split_file):
    ds_preload = AMRDataset(fake_split_file, normalize="per_sample", preload=True)
    ds_lazy = AMRDataset(fake_split_file, normalize="per_sample", preload=False)

    for i in range(len(ds_preload)):
        iq_a, mod_a, snr_a, dom_a = ds_preload[i]
        iq_b, mod_b, snr_b, dom_b = ds_lazy[i]

        torch.testing.assert_close(iq_a, iq_b)
        assert mod_a.item() == mod_b.item()
        assert snr_a.item() == pytest.approx(snr_b.item())
        assert dom_a.item() == dom_b.item()

    ds_lazy.close()


# ---------------------------------------------------------------------------
# DataLoader integration (batching works end-to-end)
# ---------------------------------------------------------------------------

def test_works_with_dataloader(fake_split_file):
    from torch.utils.data import DataLoader

    ds = AMRDataset(fake_split_file, normalize="per_sample")
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    batch = next(iter(loader))
    iq_batch, mod_batch, snr_batch, domain_batch = batch

    assert iq_batch.shape == (8, 2, SEQ_LEN)
    assert mod_batch.shape == (8,)
    assert snr_batch.shape == (8,)
    assert domain_batch.shape == (8,)


# ---------------------------------------------------------------------------
# Module-level constants sanity
# ---------------------------------------------------------------------------

def test_mod_names_length():
    assert len(MOD_NAMES) == N_MODS


def test_domain_names_length():
    assert len(DOMAIN_NAMES) == 2