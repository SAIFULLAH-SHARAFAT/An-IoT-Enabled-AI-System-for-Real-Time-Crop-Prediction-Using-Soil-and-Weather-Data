"""TabNet classifier (paper Section 3.4, Table 6): the best deep model at 92.0 % accuracy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from crop_recommendation.data import PreparedData
from crop_recommendation.evaluation import Metrics, compute_metrics
from crop_recommendation.utils import seed_everything, timer

LOGGER = logging.getLogger(__name__)

MODEL_PARAMS: dict[str, Any] = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "n_independent": 2,
    "n_shared": 2,
    "mask_type": "entmax",
}
OPTIMIZER_LR = 2e-2
FIT_PARAMS: dict[str, Any] = {
    "max_epochs": 100,
    "patience": 10,
    "batch_size": 256,
    "virtual_batch_size": 128,
    "num_workers": 0,
}


@dataclass
class TabNetResult:
    model: TabNetClassifier
    train_accuracy: float
    test_metrics: Metrics
    n_parameters: int
    model_size_mb: float
    avg_epoch_seconds: float
    epochs_run: int
    history: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "TabNet",
            "train_accuracy": self.train_accuracy,
            "test": self.test_metrics.to_dict(),
            "n_parameters": self.n_parameters,
            "model_size_mb": round(self.model_size_mb, 4),
            "avg_epoch_seconds": round(self.avg_epoch_seconds, 4),
            "epochs_run": self.epochs_run,
            "history": self.history,
        }


def fit_tabnet(
    data: PreparedData, *, seed: int, device_name: str = "auto", verbose: int = 0
) -> TabNetResult:
    """Train TabNet with the paper's configuration and early stopping on test accuracy."""
    seed_everything(seed)
    X_train, X_test = data.X_train.to_numpy(), data.X_test.to_numpy()
    model = TabNetClassifier(
        **MODEL_PARAMS,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": OPTIMIZER_LR},
        seed=seed,
        device_name=device_name,
        verbose=verbose,
    )
    with timer() as watch:
        model.fit(
            X_train,
            data.y_train,
            eval_set=[(X_train, data.y_train), (X_test, data.y_test)],
            eval_name=["train", "test"],
            eval_metric=["accuracy"],
            **FIT_PARAMS,
        )
    epochs_run = len(model.history["loss"])

    with timer() as inference:
        predictions = model.predict(X_test)
    test_metrics = compute_metrics(data.y_test, predictions, inference_seconds=inference.seconds)
    train_accuracy = float(np.mean(model.predict(X_train) == data.y_train))
    parameters = list(model.network.parameters())
    LOGGER.info("TabNet | test %s", test_metrics.summary())
    return TabNetResult(
        model=model,
        train_accuracy=train_accuracy,
        test_metrics=test_metrics,
        n_parameters=sum(p.numel() for p in parameters),
        model_size_mb=sum(p.element_size() * p.numel() for p in parameters) / 1e6,
        avg_epoch_seconds=watch.seconds / epochs_run,
        epochs_run=epochs_run,
        history={key: [float(v) for v in values] for key, values in model.history.history.items()},
    )
