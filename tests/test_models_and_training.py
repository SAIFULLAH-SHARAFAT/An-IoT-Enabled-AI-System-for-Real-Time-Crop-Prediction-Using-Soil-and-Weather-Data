from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from crop_recommendation.data import PreparedData
from crop_recommendation.evaluation import (
    compute_metrics,
    evaluate,
    plot_confusion_matrix,
    plot_roc_curves,
)
from crop_recommendation.models.classical import MODEL_SPECS, build_search, get_spec
from crop_recommendation.models.stacking import (
    BASE_LEARNERS,
    BEST_META,
    STACK_SPECS,
    build_stack,
    build_stack_search,
    get_stack_spec,
)
from crop_recommendation.training import train_classical, train_stack


def test_every_classical_spec_builds_both_variants() -> None:
    assert set(MODEL_SPECS) == {
        "dt",
        "rf",
        "nb",
        "lr",
        "svc",
        "knn",
        "gb",
        "xgb",
        "catboost",
    }
    for spec in MODEL_SPECS.values():
        assert clone(spec.default(42)) is not None
        assert clone(spec.tuned(42)) is not None


def test_recorded_hyperparameters_are_preserved() -> None:
    assert get_spec("rf").tuned(42).get_params()["n_estimators"] == 200
    knn = get_spec("knn").tuned(42).get_params()
    assert (knn["n_neighbors"], knn["weights"], knn["metric"]) == (11, "distance", "manhattan")
    gb = get_spec("gb").tuned(42).get_params()
    assert (gb["n_estimators"], gb["learning_rate"], gb["max_depth"], gb["subsample"]) == (
        150,
        0.1,
        5,
        0.8,
    )
    assert get_spec("lr").tuned(42).get_params()["C"] == 100


def test_unknown_keys_list_alternatives() -> None:
    with pytest.raises(KeyError, match="Available"):
        get_spec("nope")
    with pytest.raises(KeyError, match="Available"):
        get_stack_spec("nope")


def test_search_grids_only_name_real_parameters() -> None:
    for spec in MODEL_SPECS.values():
        if spec.grid:
            valid = spec.default(42).get_params() | spec.tuned(42).get_params()
            assert set(spec.grid) <= set(valid), spec.key
    with pytest.raises(ValueError, match="No search space"):
        build_search(MODEL_SPECS["xgb"], 42)


def test_stack_specs_match_table5() -> None:
    assert set(STACK_SPECS) == {"dt", "rf", "nb", "knn", "lr", "svc", "gb", "catboost"}
    assert BEST_META == "svc"
    assert STACK_SPECS["svc"].base_keys == ("rf", "gb", "catboost")
    assert set(STACK_SPECS["rf"].base_keys) == {"dt", "gb", "catboost"}  # Table 5
    assert STACK_SPECS["catboost"].base_keys == ("rf", "dt", "gb")


def test_stack_grids_target_the_meta_learner() -> None:
    for spec in STACK_SPECS.values():
        stack = build_stack(spec, 42)
        params = stack.get_params()
        assert set(spec.grid) <= set(params), spec.key
        assert all(key.startswith("final_estimator__") for key in spec.grid)
        assert [name for name, _ in stack.estimators] == [f"{k}_base" for k in spec.base_keys]
        assert set(spec.base_keys) <= set(BASE_LEARNERS)


def test_svc_stack_search_uses_two_fold_cv() -> None:
    search = build_stack_search(STACK_SPECS["svc"], 42)
    assert search.cv == 2
    assert search.estimator.final_estimator.probability is True


def test_train_classical_on_separable_data(synthetic_data: PreparedData) -> None:
    result = train_classical(synthetic_data, "dt", tuned=True, seed=42)
    assert result.test_metrics.accuracy > 0.95
    bundle = result.to_bundle(synthetic_data)
    assert bundle.metadata["name"].startswith("Decision Tree")
    assert result.to_dict()["test"]["n_samples"] == len(synthetic_data.y_test)


def test_train_stack_on_separable_data(synthetic_data: PreparedData) -> None:
    result = train_stack(synthetic_data, "nb", seed=42, n_jobs=1)
    assert result.test_metrics.accuracy > 0.95
    assert result.details["base_learners"] == ["rf", "gb", "catboost"]
    assert "final_estimator__var_smoothing" in result.details["best_params"]


def test_compute_metrics_and_timing(synthetic_data: PreparedData) -> None:
    perfect = compute_metrics([0, 1, 2], [0, 1, 2])
    assert perfect.accuracy == perfect.f1 == 1.0
    wrong = compute_metrics([0, 1, 2], [1, 1, 2])
    assert wrong.accuracy == pytest.approx(2 / 3)

    model = get_spec("dt").default(42).fit(synthetic_data.X_train, synthetic_data.y_train)
    metrics = evaluate(model, synthetic_data.X_test, synthetic_data.y_test)
    assert metrics.inference_seconds > 0
    assert "acc=" in metrics.summary()


def test_plots_are_written(tmp_path, synthetic_data: PreparedData) -> None:
    model = get_spec("rf").tuned(42).fit(synthetic_data.X_train, synthetic_data.y_train)
    predictions = model.predict(synthetic_data.X_test)
    cm = plot_confusion_matrix(
        synthetic_data.y_test, predictions, synthetic_data.classes, tmp_path / "cm.png"
    )
    roc = plot_roc_curves(
        synthetic_data.y_test,
        model.predict_proba(synthetic_data.X_test),
        synthetic_data.classes,
        tmp_path / "roc.png",
    )
    assert cm.stat().st_size > 0 and roc.stat().st_size > 0
    assert not list(tmp_path.glob(".*"))  # atomic writes leave no temp files
    assert np.isclose(model.predict_proba(synthetic_data.X_test).sum(axis=1), 1).all()


def test_stack_experiments_keep_their_original_base_configurations() -> None:
    dt_stack = build_stack(get_stack_spec("dt"), 42)
    bases = dict(dt_stack.estimators)
    assert bases["gb_base"].n_estimators == 100
    assert bases["catboost_base"].get_param("iterations") == 500

    rf_spec = get_stack_spec("rf")
    assert rf_spec.base_keys == ("gb", "catboost", "dt")
    assert set(rf_spec.base_search) == {"gb", "catboost", "dt"}
    assert dict(build_stack(rf_spec, 42).estimators)["gb_base"].subsample == 0.9

    assert dict(build_stack(get_stack_spec("svc"), 42).estimators)["gb_base"].n_estimators == 150
    assert dict(build_stack(get_stack_spec("catboost"), 42).estimators)["rf_base"].max_depth == 5


def test_stack_with_tuned_bases_trains(synthetic_data: PreparedData) -> None:
    result = train_stack(synthetic_data, "rf", seed=42, n_jobs=1)
    assert result.test_metrics.accuracy > 0.95
    assert result.details["base_learners"] == ["gb", "catboost", "dt"]


def test_logistic_regression_search_matches_original_protocol() -> None:
    search = build_search(get_spec("lr"), 42)
    assert search.estimator.max_iter == 500
    assert search.param_grid["penalty"] == ["l2"]
