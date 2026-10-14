"""Optuna hyper-parameter search for the deep models (TPE sampler, 20 trials in the paper)."""

from __future__ import annotations

import logging
from typing import Any

import optuna
import torch

from crop_recommendation.data import PreparedData
from crop_recommendation.models.deep.registry import DeepSpec
from crop_recommendation.models.deep.trainer import fit_deep

LOGGER = logging.getLogger(__name__)

DEFAULT_TRIALS = 20


def suggest(trial: optuna.Trial, space: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    """Draw one configuration from the declarative ``space`` of a :class:`DeepSpec`."""
    params: dict[str, Any] = {}
    for name, (kind, *args) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, args[0], args[1])
        elif kind == "int_step":
            params[name] = trial.suggest_int(name, args[0], args[1], step=args[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, args[0], args[1])
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, args[0], args[1], log=True)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(args[0]))
        else:
            raise ValueError(f"Unknown search-space kind '{kind}' for '{name}'")
    return params


def tune(
    spec: DeepSpec,
    data: PreparedData,
    *,
    seed: int,
    n_trials: int = DEFAULT_TRIALS,
    device: torch.device | None = None,
) -> optuna.Study:
    """Maximise hold-out accuracy over ``n_trials`` configurations (as in the original protocol)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest(trial, spec.space)
        return fit_deep(
            spec, data, seed=seed, device=device, overrides=overrides
        ).final_metrics.accuracy

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    LOGGER.info("%s | best accuracy %.4f with %s", spec.label, study.best_value, study.best_params)
    return study
