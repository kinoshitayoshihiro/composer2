from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PedalModel(nn.Module):
    """Conv1D + BiLSTM pedal state predictor."""

    def __init__(
        self,
        input_dim: int = 13,
        conv_channels: list[int] | None = None,
        lstm_hidden: int = 64,
        class_weight: float | None = None,
    ) -> None:
        super().__init__()
        conv_channels = conv_channels or [32, 64]
        layers = []
        in_ch = input_dim
        for ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            in_ch, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden * 2, 1)
        weight = torch.tensor(class_weight) if class_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        logits = self.fc(out).squeeze(-1)
        return logits

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)
