"""Training loop for the PyTorch models (Adam, cross-entropy, batch size 32)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from crop_recommendation.data import PreparedData
from crop_recommendation.evaluation import Metrics, compute_metrics
from crop_recommendation.models.deep.architectures import count_parameters, model_size_mb
from crop_recommendation.models.deep.registry import DeepSpec
from crop_recommendation.utils import seed_everything, timer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float


@dataclass
class DeepResult:
    """Outcome of one training run."""

    name: str
    model: nn.Module
    history: list[EpochRecord]
    final_metrics: Metrics  # test metrics after the last epoch
    best_epoch: int  # epoch with the highest test accuracy (as inspected in the notebooks)
    best_test_accuracy: float
    epoch_mean_test_accuracy: float  # mean over all epochs (how the original notebooks pooled it)
    n_parameters: int
    model_size_mb: float
    avg_epoch_seconds: float
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "test": self.final_metrics.to_dict(),
            "best_epoch": self.best_epoch,
            "best_test_accuracy": self.best_test_accuracy,
            "epoch_mean_test_accuracy": self.epoch_mean_test_accuracy,
            "n_parameters": self.n_parameters,
            "model_size_mb": round(self.model_size_mb, 4),
            "avg_epoch_seconds": round(self.avg_epoch_seconds, 4),
            "params": self.params,
            "history": [record.__dict__ for record in self.history],
        }


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@contextmanager
def inference_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Switch ``model`` to eval mode without gradients; restore the previous mode on exit."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield model
    finally:
        model.train(was_training)


def make_loaders(
    spec: DeepSpec, data: PreparedData, seed: int, batch_size: int | None = None
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Shuffled training loader (seeded) and sequential test loader."""
    size = batch_size or spec.batch_size
    train_set = TensorDataset(
        torch.from_numpy(spec.reshape(data.X_train.to_numpy())),
        torch.from_numpy(data.y_train).long(),
    )
    test_set = TensorDataset(
        torch.from_numpy(spec.reshape(data.X_test.to_numpy())),
        torch.from_numpy(data.y_test).long(),
    )
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_set, batch_size=size, shuffle=True, generator=generator),
        DataLoader(test_set, batch_size=size, shuffle=False),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, NDArray[np.int64], NDArray[np.int64]]:
    """Run one epoch over ``loader`` (training when ``optimizer`` is set)."""
    total_loss = 0.0
    targets: list[Tensor] = []
    predictions: list[Tensor] = []
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        predictions.append(outputs.argmax(dim=1).cpu())
        targets.append(labels.cpu())
    return (
        total_loss / len(loader),
        torch.cat(targets).numpy().astype(np.int64),
        torch.cat(predictions).numpy().astype(np.int64),
    )


def fit_deep(
    spec: DeepSpec,
    data: PreparedData,
    *,
    seed: int,
    device: torch.device | None = None,
    overrides: dict[str, Any] | None = None,
) -> DeepResult:
    """Train ``spec`` with its recorded hyper-parameters (optionally overridden)."""
    params = {**spec.params, **(overrides or {})}
    device = device or torch.device("cpu")
    seed_everything(seed)
    train_loader, test_loader = make_loaders(spec, data, seed)

    model = spec.build(data.n_classes, params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.CrossEntropyLoss()

    history: list[EpochRecord] = []
    epoch_seconds: list[float] = []
    last_test_pred = np.empty(0, dtype=np.int64)
    last_test_true = np.empty(0, dtype=np.int64)
    for epoch in range(1, params["epochs"] + 1):
        with timer() as watch:
            model.train()
            train_loss, train_true, train_pred = _run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            with inference_mode(model):
                test_loss, last_test_true, last_test_pred = _run_epoch(
                    model, test_loader, criterion, device
                )
        epoch_seconds.append(watch.seconds)
        record = EpochRecord(
            epoch,
            train_loss,
            float((train_pred == train_true).mean()),
            test_loss,
            float((last_test_pred == last_test_true).mean()),
        )
        history.append(record)
        LOGGER.debug(
            "%s epoch %d/%d loss=%.4f train_acc=%.4f test_acc=%.4f",
            spec.label,
            epoch,
            params["epochs"],
            record.train_loss,
            record.train_accuracy,
            record.test_accuracy,
        )

    best = max(history, key=lambda r: r.test_accuracy)
    final_metrics = compute_metrics(last_test_true, last_test_pred)
    LOGGER.info(
        "%s | final test acc=%.4f (best epoch %d: %.4f)",
        spec.label,
        final_metrics.accuracy,
        best.epoch,
        best.test_accuracy,
    )
    return DeepResult(
        name=spec.label,
        model=model,
        history=history,
        final_metrics=final_metrics,
        best_epoch=best.epoch,
        best_test_accuracy=best.test_accuracy,
        epoch_mean_test_accuracy=float(np.mean([r.test_accuracy for r in history])),
        n_parameters=count_parameters(model),
        model_size_mb=model_size_mb(model),
        avg_epoch_seconds=float(np.mean(epoch_seconds)),
        params=params,
    )
