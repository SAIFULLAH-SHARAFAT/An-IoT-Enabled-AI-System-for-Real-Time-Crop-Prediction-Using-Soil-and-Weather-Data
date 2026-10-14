"""Deep-model registry: final architectures (Table 6) and their Optuna search spaces.

CNN, CNN-LSTM and CNN-BiLSTM settings are the best trials recorded in the original
notebooks. The LSTM notebook is not part of the repository: its architecture (2 layers x 96
hidden units, 117,334 parameters) is taken from the paper; learning rate and epoch count are
reconstructed (30 epochs per Fig. 19) and flagged as such below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch import nn

from crop_recommendation.models.deep.architectures import (
    CropCNN,
    CropCNNBiLSTM,
    CropCNNLSTM,
    CropLSTM,
    to_grid,
    to_sequence,
)

Reshape = Callable[[NDArray[np.float64]], NDArray[np.float32]]
Builder = Callable[[int, dict[str, Any]], nn.Module]


@dataclass(frozen=True)
class DeepSpec:
    """Everything needed to build, train and tune one deep model."""

    key: str
    label: str
    reshape: Reshape
    build: Builder
    params: dict[str, Any]  # final hyper-parameters, including ``lr`` and ``epochs``
    space: dict[str, tuple[Any, ...]]  # Optuna search space: name -> (kind, *args)
    reconstructed: bool = False  # True when not recoverable from the shipped notebooks
    batch_size: int = 32


def _grid(height: int, width: int) -> Reshape:
    return lambda features: to_grid(features, height, width)


def _build_cnn(num_classes: int, p: dict[str, Any]) -> nn.Module:
    return CropCNN((1, 2, 4), num_classes, p["num_layers"], p["init_channels"], p["kernel_size"])


def _build_cnn_lstm(num_classes: int, p: dict[str, Any]) -> nn.Module:
    return CropCNNLSTM(
        (1, 3, 3),
        num_classes,
        p["num_layers"],
        p["init_channels"],
        p["kernel_size"],
        p["lstm_hidden_size"],
        p["lstm_layers"],
    )


def _build_cnn_bilstm(num_classes: int, p: dict[str, Any]) -> nn.Module:
    return CropCNNBiLSTM(
        (1, 2, 4),
        num_classes,
        p["num_layers"],
        p["init_channels"],
        p["kernel_size"],
        p["lstm_hidden_size"],
        p["lstm_layers"],
    )


def _build_lstm(num_classes: int, p: dict[str, Any]) -> nn.Module:
    return CropLSTM(8, num_classes, p["hidden_size"], p["num_layers"], p["dropout"])


DEEP_SPECS: dict[str, DeepSpec] = {
    spec.key: spec
    for spec in (
        DeepSpec(
            key="cnn",
            label="CNN",
            reshape=_grid(2, 4),
            build=_build_cnn,
            params={
                "num_layers": 2,
                "init_channels": 32,
                "kernel_size": 2,
                "lr": 0.008133728243728402,
                "epochs": 25,
            },
            space={
                "num_layers": ("int", 1, 3),
                "init_channels": ("categorical", (16, 32, 64)),
                "kernel_size": ("int", 2, 3),
                "lr": ("float_log", 1e-4, 1e-2),
                "epochs": ("int", 10, 30),
            },
        ),
        DeepSpec(
            key="lstm",
            label="LSTM",
            reshape=to_sequence,
            build=_build_lstm,
            params={"hidden_size": 96, "num_layers": 2, "dropout": 0.0, "lr": 0.005, "epochs": 30},
            space={
                "hidden_size": ("int", 32, 128),
                "num_layers": ("int", 1, 3),
                "dropout": ("float", 0.0, 0.5),
                "lr": ("float_log", 1e-4, 1e-2),
                "epochs": ("int", 10, 30),
            },
            reconstructed=True,
        ),
        DeepSpec(
            key="cnn_lstm",
            label="CNN-LSTM",
            reshape=_grid(3, 3),
            build=_build_cnn_lstm,
            params={
                "num_layers": 4,
                "init_channels": 16,
                "kernel_size": 3,
                "lstm_hidden_size": 64,
                "lstm_layers": 1,
                "lr": 0.0033640580511635804,
                "epochs": 26,
            },
            space={
                "num_layers": ("int", 1, 5),
                "init_channels": ("int_step", 16, 64, 16),
                "kernel_size": ("int", 2, 3),
                "lstm_hidden_size": ("int_step", 64, 256, 64),
                "lstm_layers": ("int", 1, 3),
                "lr": ("float_log", 1e-4, 1e-2),
                "epochs": ("int", 5, 30),
            },
        ),
        DeepSpec(
            key="cnn_bilstm",
            label="CNN-BiLSTM",
            reshape=_grid(2, 4),
            build=_build_cnn_bilstm,
            params={
                "num_layers": 3,
                "init_channels": 16,
                "kernel_size": 2,
                "lstm_hidden_size": 99,
                "lstm_layers": 1,
                "lr": 0.0008564468987587281,
                "epochs": 23,
            },
            space={
                "num_layers": ("int", 1, 3),
                "init_channels": ("categorical", (16, 32, 64)),
                "kernel_size": ("int", 2, 3),
                "lstm_hidden_size": ("int", 64, 256),
                "lstm_layers": ("int", 1, 3),
                "lr": ("float_log", 1e-4, 1e-2),
                "epochs": ("int", 10, 30),
            },
        ),
    )
}


def get_deep_spec(key: str) -> DeepSpec:
    try:
        return DEEP_SPECS[key]
    except KeyError:
        raise KeyError(f"Unknown deep model '{key}'. Available: {', '.join(DEEP_SPECS)}") from None
