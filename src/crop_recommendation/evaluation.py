"""Metrics and diagnostic plots (paper Section 3.6, Eq. 1-3, Fig. 15-16)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from crop_recommendation.utils import atomic_path, timer

LOGGER = logging.getLogger(__name__)


class Classifier(Protocol):
    """Minimal estimator interface used for evaluation."""

    def predict(self, X: Any) -> Any: ...


@dataclass(frozen=True)
class Metrics:
    """Weighted-average classification metrics plus inference latency."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    n_samples: int
    inference_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"acc={self.accuracy:.3f} precision={self.precision:.3f} "
            f"recall={self.recall:.3f} f1={self.f1:.3f} "
            f"inference={self.inference_seconds:.4f}s ({self.n_samples} samples)"
        )


def compute_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, *, inference_seconds: float = 0.0
) -> Metrics:
    """Accuracy and weighted precision / recall / F1 (Eq. 1-3 averaged over classes)."""
    y_true_arr = np.asarray(y_true)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred, average="weighted", zero_division=0
    )
    return Metrics(
        accuracy=float(accuracy_score(y_true_arr, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        n_samples=len(y_true_arr),
        inference_seconds=inference_seconds,
    )


def evaluate(model: Classifier, X: pd.DataFrame, y: ArrayLike) -> Metrics:
    """Predict ``X`` (timing the call, as in the paper's inference-time columns) and score it."""
    with timer() as watch:
        predictions = model.predict(X)
    return compute_metrics(y, np.asarray(predictions).ravel(), inference_seconds=watch.seconds)


def plot_confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, labels: Sequence[str], path: Path | str
) -> Path:
    """Save a confusion matrix figure (cf. Fig. 16)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=list(labels),
        cmap="Blues",
        xticks_rotation="vertical",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    with atomic_path(path) as tmp:
        fig.savefig(tmp, dpi=300, format=path.suffix.lstrip(".") or "png")
    plt.close(fig)
    return path


def plot_roc_curves(
    y_true: ArrayLike, probabilities: NDArray[np.float64], labels: Sequence[str], path: Path | str
) -> Path:
    """Save one-vs-rest ROC curves for every class (cf. Fig. 15)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    binarized = label_binarize(np.asarray(y_true), classes=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 8))
    for index, name in enumerate(labels):
        fpr, tpr, _ = roc_curve(binarized[:, index], probabilities[:, index])
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(loc="lower right", fontsize="x-small", ncol=2)
    fig.tight_layout()
    with atomic_path(path) as tmp:
        fig.savefig(tmp, dpi=300, format=path.suffix.lstrip(".") or "png")
    plt.close(fig)
    return path
