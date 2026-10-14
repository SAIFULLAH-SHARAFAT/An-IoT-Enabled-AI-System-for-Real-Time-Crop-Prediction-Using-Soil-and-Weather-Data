from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crop_recommendation.config import (
    DATASET_ENV_VAR,
    EXPECTED_CLASSES,
    EXPECTED_ROWS,
    FEATURES,
    TARGET,
    default_dataset_path,
)
from crop_recommendation.data import (
    DataValidationError,
    clean,
    load_dataset,
    prepare,
    validate_schema,
)

#: SHA-256 of the dataset as published with the paper. The file must never change.
DATASET_SHA256 = "aab47fc023216760348ce51f77ac49cc1e9e89a27447068a8003ed30e429b01a"


def test_feature_order_matches_paper() -> None:
    assert FEATURES == (
        "N",
        "P",
        "K",
        "pH",
        "Temp(°C)",
        "Humidity(%)",
        "Moisture(%)",
        "Rainfall(cm)",
    )


def test_dataset_is_unmodified(dataset_path: Path) -> None:
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == DATASET_SHA256


def test_dataset_matches_paper_description(dataset_path: Path) -> None:
    frame = load_dataset(dataset_path)
    assert frame.shape == (EXPECTED_ROWS, len(FEATURES) + 1)
    counts = frame[TARGET].value_counts()
    assert len(counts) == EXPECTED_CLASSES
    assert set(counts) == {150}
    assert frame.isna().sum().sum() == 0
    assert not frame.duplicated().any()
    # Table 3 of the paper
    assert frame["N"].mean() == pytest.approx(223.728, abs=1e-3)
    assert frame["pH"].min() == pytest.approx(3.3)


def test_default_dataset_path_honours_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(DATASET_ENV_VAR, str(tmp_path / "custom.xlsx"))
    assert default_dataset_path() == tmp_path / "custom.xlsx"


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.xlsx")


def test_validate_schema_reports_missing_columns(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_schema(synthetic_frame.drop(columns=["pH"]))


def test_validate_schema_rejects_non_numeric_features(synthetic_frame: pd.DataFrame) -> None:
    broken = synthetic_frame.astype({"N": object})
    broken.loc[0, "N"] = "high"
    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate_schema(broken)


def test_clean_removes_nulls_and_duplicates(synthetic_frame: pd.DataFrame) -> None:
    dirty = pd.concat([synthetic_frame, synthetic_frame.iloc[[0]]], ignore_index=True)
    dirty.loc[1, "K"] = np.nan
    cleaned = clean(dirty)
    assert len(cleaned) == len(synthetic_frame) - 1
    assert cleaned.isna().sum().sum() == 0


def test_prepare_splits_stratified_80_20(synthetic_frame: pd.DataFrame) -> None:
    data = prepare(synthetic_frame)
    assert len(data.X_train) + len(data.X_test) == len(synthetic_frame)
    assert len(data.X_test) == pytest.approx(0.2 * len(synthetic_frame), abs=1)
    assert set(np.bincount(data.y_test)) == {len(data.X_test) // data.n_classes}
    assert list(data.X_train.columns) == list(FEATURES)


def test_prepare_scales_to_unit_range_and_round_trips_labels(synthetic_frame: pd.DataFrame) -> None:
    data = prepare(synthetic_frame)
    combined = pd.concat([data.X_train, data.X_test])
    assert combined.min().min() == pytest.approx(0.0)
    assert combined.max().max() == pytest.approx(1.0)
    assert set(data.classes) == set(synthetic_frame[TARGET])
    assert data.label_encoder.inverse_transform([0])[0] == data.classes[0]


def test_prepare_is_deterministic(synthetic_frame: pd.DataFrame) -> None:
    first, second = prepare(synthetic_frame, seed=7), prepare(synthetic_frame, seed=7)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    other = prepare(synthetic_frame, seed=8)
    assert not first.X_test.index.equals(other.X_test.index)


def test_real_dataset_split_sizes(dataset_path: Path) -> None:
    data = prepare(load_dataset(dataset_path))
    assert data.X_train.shape == (2640, 8)
    assert data.X_test.shape == (660, 8)
    assert data.n_classes == EXPECTED_CLASSES
