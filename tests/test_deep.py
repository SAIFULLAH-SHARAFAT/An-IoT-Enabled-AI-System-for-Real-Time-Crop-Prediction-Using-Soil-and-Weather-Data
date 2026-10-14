from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crop_recommendation.data import PreparedData  # noqa: E402
from crop_recommendation.models.deep.architectures import (  # noqa: E402
    count_parameters,
    model_size_mb,
    to_grid,
    to_sequence,
)
from crop_recommendation.models.deep.registry import DEEP_SPECS, get_deep_spec  # noqa: E402
from crop_recommendation.models.deep.trainer import fit_deep, inference_mode  # noqa: E402
from crop_recommendation.models.deep.tuning import suggest  # noqa: E402
from crop_recommendation.reference import TABLE6_DEEP  # noqa: E402

pytestmark = pytest.mark.deep

PAPER_CLASSES = 22


@pytest.mark.parametrize("key", ["cnn", "lstm", "cnn_lstm", "cnn_bilstm"])
def test_parameter_counts_match_table6(key: str) -> None:
    """The reconstructed architectures have exactly the trainable parameters the paper reports."""
    spec = get_deep_spec(key)
    model = spec.build(PAPER_CLASSES, spec.params)
    assert count_parameters(model) == TABLE6_DEEP[key][1]


def test_model_sizes_are_in_the_reported_range() -> None:
    sizes = {k: model_size_mb(s.build(PAPER_CLASSES, s.params)) for k, s in DEEP_SPECS.items()}
    assert sizes["cnn"] == pytest.approx(0.08, abs=0.01)
    assert sizes["cnn_lstm"] == pytest.approx(0.09, abs=0.01)
    assert sizes["cnn_bilstm"] == pytest.approx(2.12, abs=0.01)


def test_reshaping_helpers() -> None:
    import numpy as np

    rows = np.arange(16, dtype=float).reshape(2, 8)
    grid = to_grid(rows, 3, 3)
    assert grid.shape == (2, 1, 3, 3) and grid.dtype == np.float32
    assert grid.reshape(2, -1)[0, -1] == 0.0  # zero padding
    assert to_grid(rows, 2, 4).shape == (2, 1, 2, 4)
    assert to_sequence(rows).shape == (2, 1, 8)
    with pytest.raises(ValueError, match="holds"):
        to_grid(rows, 2, 2)


def test_deep_training_smoke(synthetic_data: PreparedData) -> None:
    spec = get_deep_spec("cnn")
    result = fit_deep(spec, synthetic_data, seed=42, overrides={"epochs": 8, "lr": 0.01})
    assert len(result.history) == 8
    assert result.final_metrics.accuracy > 0.8
    assert result.best_test_accuracy >= result.final_metrics.accuracy
    assert result.to_dict()["n_parameters"] == result.n_parameters


def test_deep_training_is_reproducible(synthetic_data: PreparedData) -> None:
    spec = get_deep_spec("lstm")
    first = fit_deep(spec, synthetic_data, seed=1, overrides={"epochs": 2})
    second = fit_deep(spec, synthetic_data, seed=1, overrides={"epochs": 2})
    assert [r.train_loss for r in first.history] == [r.train_loss for r in second.history]


def test_inference_mode_restores_training_state() -> None:
    model = torch.nn.Linear(2, 2)
    model.train()
    with inference_mode(model):
        assert not model.training
        assert not torch.is_grad_enabled()
    assert model.training


def test_search_spaces_are_well_formed() -> None:
    optuna = pytest.importorskip("optuna")
    for spec in DEEP_SPECS.values():
        trial = optuna.trial.FixedTrial(_fixed_values(spec.space))
        params = suggest(trial, spec.space)
        assert set(params) == set(spec.space)
        assert 1e-4 <= params["lr"] <= 1e-2


def _fixed_values(space: dict[str, tuple[object, ...]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, (kind, *args) in space.items():
        if kind == "categorical":
            values[name] = args[0][0]  # type: ignore[index]
        elif kind in {"int", "int_step", "float", "float_log"}:
            values[name] = args[0]
    return values
