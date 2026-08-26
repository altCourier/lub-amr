"""
amr_dataset.py

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

"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

MOD_NAMES = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM']
DOMAIN_NAMES = ['Rma', 'Umi']  # 0, 1

class AMRDataset(Dataset):
    """
    PyTorch Dataset over a single split file (train.h5 / val.h5 / test.h5).

    Parameters
    ----------
    h5_path : str
        Path to the split file.

    normalize : str or None
        Normalization strategy applied to each (2, 1024) I/Q sample:

          - "per_sample": scale each sample to unit average energy
                           (E[|x|^2] = 1 across the 1024 symbols). Standard choice
                           for AMR since it removes absolute amplitude as a
                           trivial shortcut while preserving relative constellation
                           shape.

          - "global":     scale by a fixed (mean, std) computed once across the
                           whole dataset and passed in via `global_stats`.

          - None:         no normalization, raw I/Q values as-is.

    global_stats : tuple(float, float) or None
        (mean, std) to use when normalize="global". Required in that case.
        Compute once on the training set and reuse for val/test to avoid leakage.

    preload : bool
        If True, reads the entire Data/Mods/SNRs/Domain arrays into RAM on init.
        If False, keeps the file handle open and reads rows
        lazily on __getitem__.

    """

    def __init__(self, h5_path, normalize="per_sample",
             global_stats=None, preload=True, domain_filter=None):
        """
        ...
        domain_filter : int or None
            0 = Rma-only, 1 = Umi-only, None = pooled (no filtering).
            Filtering is applied on load, before normalization, so per-sample
            stats and __len__ reflect only the selected domain's rows.
        """

        self.h5_path       = h5_path
        self.normalize     = normalize
        self.preload       = preload
        self.domain_filter = domain_filter

        if normalize == "global" and global_stats is None:
            raise ValueError(
                "normalize='global' requires global_stats=(mean, std), "
                "computed on the training set and passed explicitly."
            )

        self.global_stats = global_stats

        if self.preload:

            with h5py.File(self.h5_path, "r") as f:
                data   = f["Data"][:]
                mods   = f["Mods"][:]
                snrs   = f["SNRs"][:]
                domain = f["Domain"][:]

            if domain_filter is not None:
                mask = domain == domain_filter
                data, mods, snrs, domain = data[mask], mods[mask], snrs[mask], domain[mask]

            self.data, self.mods, self.snrs, self.domain = data, mods, snrs, domain

            self._file   = None
            self.indices = None  # not needed in preload mode, rows already filtered

        else:

            self._file  = h5py.File(self.h5_path, "r")
            self.data   = self._file["Data"]
            self.mods   = self._file["Mods"]
            self.snrs   = self._file["SNRs"]
            self.domain = self._file["Domain"]

            if domain_filter is not None:
                domain_full  = self.domain[:]
                self.indices = np.where(domain_full == domain_filter)[0]

            else:
                self.indices = None  # identity mapping, use idx directly

        self.n = self.data.shape[0] if self.indices is None else len(self.indices)

    def __len__(self):

        return self.n

    def _to_iq_tensor(self, complex_row):
        """
        complex64 (1024,) -> float32 tensor (2, 1024): channel 0 = I, channel 1 = Q
        """

        iq = np.stack([complex_row.real, complex_row.imag], axis=0).astype(np.float32)


        if self.normalize == "per_sample":
            energy = np.mean(iq[0] ** 2 + iq[1] ** 2)
            iq = iq / np.sqrt(energy + 1e-12)

        elif self.normalize == "global":
            mean, std = self.global_stats
            iq = (iq - mean) / (std + 1e-12)

        # normalize is None -> leave as-is

        return torch.from_numpy(iq)

    def _row(self, idx):
        """
        Resolve a dataset-local idx to the underlying h5 row index.
        """

        return idx if self.indices is None else self.indices[idx]

    def __getitem__(self, idx):
        row = self._row(idx)

        complex_row = self.data[row]
        iq_tensor   = self._to_iq_tensor(complex_row)

        mod_label = torch.tensor(np.argmax(self.mods[row]), dtype = torch.long)
        snr       = torch.tensor(self.snrs[row], dtype = torch.float32)
        domain    = torch.tensor(self.domain[row], dtype = torch.long)

        return iq_tensor, mod_label, snr, domain

    def close(self):
        """
        Close the underlying h5 file handle if opened lazily (preload=False).
        """

        if self._file is not None:
            self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def compute_global_stats(h5_path):
    """
    Helper to compute (mean, std) of raw I/Q values across an entire split file,
    for use with normalize="global". Run this ONCE on the training set only,
    then reuse the same (mean, std) for val/test to avoid data leakage.
    """

    with h5py.File(h5_path, "r") as f:
        data = f["Data"][:]

    iq = np.stack([data.real, data.imag], axis=0).astype(np.float32)

    return float(iq.mean()), float(iq.std())
