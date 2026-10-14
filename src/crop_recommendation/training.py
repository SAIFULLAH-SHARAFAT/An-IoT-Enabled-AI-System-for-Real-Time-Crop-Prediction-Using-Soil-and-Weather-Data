"""Training entry points for the classical and stacking pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator, clone

from crop_recommendation.data import PreparedData
from crop_recommendation.evaluation import Metrics, evaluate
from crop_recommendation.models.classical import build_search, get_spec
from crop_recommendation.models.stacking import build_stack_search, get_stack_spec, tune_bases
from crop_recommendation.persistence import ModelBundle
from crop_recommendation.utils import seed_everything, timer

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """A fitted estimator with its hold-out metrics and (optional) search diagnostics."""

    name: str
    estimator: BaseEstimator
    train_metrics: Metrics
    test_metrics: Metrics
    fit_seconds: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train": self.train_metrics.to_dict(),
            "test": self.test_metrics.to_dict(),
            "fit_seconds": self.fit_seconds,
            **self.details,
        }

    def to_bundle(self, data: PreparedData) -> ModelBundle:
        return ModelBundle(
            model=self.estimator,
            scaler=data.scaler,
            label_encoder=data.label_encoder,
            metadata={"name": self.name, "test_accuracy": self.test_metrics.accuracy},
        )


def _finish(
    name: str,
    estimator: BaseEstimator,
    data: PreparedData,
    fit_seconds: float,
    details: dict[str, Any] | None = None,
) -> TrainingResult:
    train_metrics = evaluate(estimator, data.X_train, data.y_train)
    test_metrics = evaluate(estimator, data.X_test, data.y_test)
    LOGGER.info("%s | test %s", name, test_metrics.summary())
    return TrainingResult(name, estimator, train_metrics, test_metrics, fit_seconds, details or {})


def train_classical(
    data: PreparedData, key: str, *, tuned: bool = True, seed: int
) -> TrainingResult:
    """Fit one classifier with its default (Table 4) or tuned (Fig. 13) hyper-parameters."""
    spec = get_spec(key)
    seed_everything(seed)
    estimator = clone(spec.tuned(seed) if tuned else spec.default(seed))
    with timer() as watch:
        estimator.fit(data.X_train, data.y_train)
    suffix = "tuned" if tuned else "default"
    return _finish(f"{spec.label} ({suffix})", estimator, data, watch.seconds)


def search_classical(
    data: PreparedData, key: str, *, seed: int, n_jobs: int | None = -1
) -> TrainingResult:
    """Run the original hyper-parameter search for ``key`` and return the best estimator."""
    spec = get_spec(key)
    seed_everything(seed)
    search = build_search(spec, seed, n_jobs=n_jobs)
    with timer() as watch:
        search.fit(data.X_train, data.y_train)
    details = {
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_accuracy": float(search.best_score_),
    }
    return _finish(f"{spec.label} (searched)", search.best_estimator_, data, watch.seconds, details)


def train_stack(
    data: PreparedData, meta_key: str, *, seed: int, n_jobs: int | None = -1
) -> TrainingResult:
    """Fit a stacking ensemble (Algorithm 1) with ``meta_key`` as the meta-learner."""
    spec = get_stack_spec(meta_key)
    seed_everything(seed)
    with timer() as watch:
        estimators = (
            tune_bases(spec, seed, data.X_train, data.y_train, n_jobs=n_jobs)
            if spec.base_search
            else None
        )
        search = build_stack_search(spec, seed, n_jobs=n_jobs, estimators=estimators)
        search.fit(data.X_train, data.y_train)
    details = {
        "base_learners": list(spec.base_keys),
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_accuracy": float(search.best_score_),
    }
    return _finish(
        f"Stacking [{'+'.join(spec.base_keys)}] -> {spec.label}",
        search.best_estimator_,
        data,
        watch.seconds,
        details,
    )
