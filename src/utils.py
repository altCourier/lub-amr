"""
utils.py

Small shared helpers used across train.py, notebooks, and eval scripts.
"""

import torch


def get_device() -> torch.device:
    """Returns cuda if available, else cpu."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
