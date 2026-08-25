"""
models.py

Model architectures for AMR.

Every model here takes input of shape: 
(batch, 2, 1024) -- the (I, Q) tensor produced by AMRDataset (see amr_dataset.py) 
                 -- and outputs raw logits over
                 
the 5 modulation classes (see MOD_NAMES in amr_dataset.py).

"""

import torch
import torch.nn as nn

N_CLASSES = 5  # BPSK, QPSK, 16QAM, 64QAM, 256QAM

class BaselineCNN(nn.Module):
    """

    Small 1D CNN baseline for AMR, in the spirit of the original RadioML
    CNN architectures (O'Shea et al.). 
    
    Three conv blocks (conv -> batchnorm -> relu -> maxpool), 
    followed by global average pooling and a small fully-connected classification head.

    Input:  (batch, 2, 1024)  -- I/Q channels
    Output: (batch, n_classes) -- raw logits (use CrossEntropyLoss, not softmax)

    """

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3):

        super().__init__()

        self.conv_block = nn.Sequential(

            # block 1: (2, 1024) -> (64, 512)
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # block 2: (64, 512) -> (128, 256)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            # block 3: (128, 256) -> (128, 128)
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # global average pool over the remaining sequence length,
        # so the classifier head doesn't depend on the exact input length
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch, 2, 1024)
        
        x = self.conv_block(x)          # (batch, 128, 128)
        x = self.global_pool(x)         # (batch, 128, 1)
        x = x.squeeze(-1)               # (batch, 128)

        logits = self.classifier(x)     # (batch, n_classes)

        return logits
