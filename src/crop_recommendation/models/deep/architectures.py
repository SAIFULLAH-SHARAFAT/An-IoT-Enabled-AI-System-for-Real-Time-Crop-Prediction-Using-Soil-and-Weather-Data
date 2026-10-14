"""PyTorch architectures for the deep-learning comparison (paper Section 3.4, Table 6).

Tabular rows are either reshaped into a small single-channel grid (CNN family) or fed as a
length-1 sequence (LSTM), exactly as in the original experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F


def to_grid(features: NDArray[np.float64], height: int, width: int) -> NDArray[np.float32]:
    """Zero-pad rows to ``height * width`` values and reshape to ``(N, 1, height, width)``."""
    n_rows, n_features = features.shape
    cells = height * width
    if n_features > cells:
        raise ValueError(f"A {height}x{width} grid holds {cells} values, got {n_features} features")
    padded = np.pad(features, ((0, 0), (0, cells - n_features)), mode="constant")
    return padded.reshape(n_rows, 1, height, width).astype(np.float32)


def to_sequence(features: NDArray[np.float64]) -> NDArray[np.float32]:
    """Reshape rows to ``(N, 1, n_features)``: one time step carrying every feature."""
    return features.reshape(features.shape[0], 1, -1).astype(np.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Parameter storage in megabytes (decimal), as reported in Table 6."""
    return sum(p.element_size() * p.numel() for p in model.parameters()) / 1e6


class CropCNN(nn.Module):
    """Stack of ``Conv2d(padding=1) + ReLU`` layers followed by a linear classifier."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        num_layers: int,
        init_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(1 if i == 0 else init_channels, init_channels, kernel_size, padding=1)
            for i in range(num_layers)
        )
        self.fc = nn.Linear(_flattened_size(self.convs, input_shape), num_classes)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x))
        logits: Tensor = self.fc(x.flatten(start_dim=1))
        return logits


class CropCNNLSTM(nn.Module):
    """Conv + max-pool blocks (channels double per block) feeding an LSTM classifier.

    Blocks stop being added once the feature map shrinks to a single row or column.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        num_layers: int,
        init_channels: int,
        kernel_size: int,
        lstm_hidden_size: int,
        lstm_layers: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        in_channels, channels = 1, init_channels
        height, width = input_shape[1], input_shape[2]
        padding = (kernel_size - 1) // 2
        for _ in range(num_layers):
            self.convs.append(nn.Conv2d(in_channels, channels, kernel_size, padding=padding))
            in_channels, channels = channels, channels * 2
            height = (height - kernel_size + 2 * padding + 1) // 2
            width = (width - kernel_size + 2 * padding + 1) // 2
            if height <= 1 or width <= 1:
                break
        self.lstm = nn.LSTM(
            input_size=in_channels * height * width,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = F.max_pool2d(F.relu(conv(x)), 2)
        sequence, _ = self.lstm(x.reshape(x.size(0), 1, -1))
        logits: Tensor = self.fc(sequence[:, -1, :])
        return logits


class CropCNNBiLSTM(nn.Module):
    """``Conv2d(padding=1)`` layers feeding a bidirectional LSTM classifier."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        num_layers: int,
        init_channels: int,
        kernel_size: int,
        lstm_hidden_size: int,
        lstm_layers: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(1 if i == 0 else init_channels, init_channels, kernel_size, padding=1)
            for i in range(num_layers)
        )
        self.lstm = nn.LSTM(
            input_size=_flattened_size(self.convs, input_shape),
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x))
        sequence, _ = self.lstm(x.flatten(start_dim=1).unsqueeze(1))
        logits: Tensor = self.fc(sequence[:, -1, :])
        return logits


class CropLSTM(nn.Module):
    """Stacked LSTM over a length-1 sequence of the eight features."""

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        sequence, _ = self.lstm(x)
        logits: Tensor = self.fc(sequence[:, -1, :])
        return logits


def _flattened_size(convs: nn.ModuleList, input_shape: tuple[int, int, int]) -> int:
    """Number of features after running a dummy input through ``Conv2d + ReLU`` layers."""
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)
        for conv in convs:
            x = F.relu(conv(x))
    return int(x.numel())
