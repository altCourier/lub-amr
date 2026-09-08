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

class CNNLSTM(nn.Module):
    """

    CNN + LSTM variant for AMR. Same convolutional backbone as BaselineCNN,
    but replaces global-average-pooling with an LSTM that reads the (128, 128)
    feature sequence step by step before classifying.

    Rationale: BaselineCNN's global-average-pool collapses the sequence
    dimension by averaging, which discards *where* along the sequence things
    happened. For high-order QAM (16/64/256), the classes differ mainly by
    constellation density -- a subtler, more structure-dependent signal than
    the coarse energy/spread differences that separate BPSK/QPSK. This variant
    tests whether preserving sequence structure through an LSTM (instead of
    averaging it away) helps recover that signal.

    Input:  (batch, 2, 1024)   -- I/Q channels
    Output: (batch, n_classes) -- raw logits (use CrossEntropyLoss, not softmax)

    """

    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3,
                 lstm_hidden: int = 128, lstm_layers: int = 1, bidirectional: bool = True):

        super().__init__()

        # identical to BaselineCNN's conv_block: (2, 1024) -> (128, 128)
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

        # reads the 128-step feature sequence (each step: 128-dim) instead of
        # averaging it away. batch_first=True to match (batch, seq, feature).
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch, 2, 1024)

        x = self.conv_block(x)              # (batch, 128, 128) = (batch, channels, seq_len)
        x = x.transpose(1, 2)               # (batch, seq_len, channels) -- LSTM wants features last

        lstm_out, (h_n, c_n) = self.lstm(x) # lstm_out: (batch, seq_len, lstm_out_dim)

        # use the last time step's output (both directions already concatenated
        # by PyTorch when bidirectional=True) rather than averaging over time,
        # so temporal position information isn't discarded the way avg-pool does
        if self.lstm.bidirectional:
            last_step = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, 2*hidden)
        else:
            last_step = h_n[-1]                                # (batch, hidden)

        logits = self.classifier(last_step) # (batch, n_classes)

        return logits
