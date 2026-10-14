from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from crop_recommendation.config import FEATURES
from crop_recommendation.inference import (
    PREDICTION_COLUMN,
    predict_file,
    predict_frame,
    read_readings,
)
from crop_recommendation.persistence import ModelBundle, write_json
from crop_recommendation.utils import atomic_path, timer


def test_bundle_round_trip(
    tmp_path: Path, synthetic_bundle: ModelBundle, synthetic_frame: pd.DataFrame
) -> None:
    path = synthetic_bundle.save(tmp_path / "nested" / "model.joblib")
    loaded = ModelBundle.load(path)
    sample = synthetic_frame.iloc[::20][list(FEATURES)]
    assert list(loaded.predict(sample)) == list(synthetic_bundle.predict(sample))
    assert not list(path.parent.glob(".*"))


def test_predict_recovers_known_crops(
    synthetic_bundle: ModelBundle, synthetic_frame: pd.DataFrame
) -> None:
    sample = synthetic_frame.iloc[::10]
    assert list(synthetic_bundle.predict(sample)) == list(sample["Crop"])


def test_predict_requires_feature_columns(synthetic_bundle: ModelBundle) -> None:
    with pytest.raises(ValueError, match="Missing feature columns"):
        synthetic_bundle.predict(pd.DataFrame({"N": [1.0]}))


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ModelBundle.load(tmp_path / "missing.joblib")
    wrong = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a bundle"}, wrong)
    with pytest.raises(TypeError, match="ModelBundle"):
        ModelBundle.load(wrong)


def test_legacy_three_file_layout(
    tmp_path: Path, synthetic_bundle: ModelBundle, synthetic_frame: pd.DataFrame
) -> None:
    paths = {name: tmp_path / f"{name}.pkl" for name in ("model", "scaler", "encoder")}
    joblib.dump(synthetic_bundle.model, paths["model"])
    joblib.dump(synthetic_bundle.scaler, paths["scaler"])
    joblib.dump(synthetic_bundle.label_encoder, paths["encoder"])
    legacy = ModelBundle.from_legacy_files(
        model_path=paths["model"], scaler_path=paths["scaler"], encoder_path=paths["encoder"]
    )
    sample = synthetic_frame.iloc[::25]
    assert list(legacy.predict(sample)) == list(synthetic_bundle.predict(sample))
    with pytest.raises(FileNotFoundError):
        ModelBundle.from_legacy_files(
            model_path=paths["model"],
            scaler_path=tmp_path / "nope.pkl",
            encoder_path=paths["encoder"],
        )


def test_predict_file_csv_and_xlsx(
    tmp_path: Path, synthetic_bundle: ModelBundle, synthetic_frame: pd.DataFrame
) -> None:
    readings = synthetic_frame.iloc[::30][list(FEATURES)]
    csv_path, xlsx_path = tmp_path / "in.csv", tmp_path / "in.xlsx"
    readings.to_csv(csv_path, index=False, encoding="utf-8")
    readings.to_excel(xlsx_path, index=False)

    out = tmp_path / "out" / "predictions.csv"
    result = predict_file(synthetic_bundle, csv_path, out)
    assert PREDICTION_COLUMN in result.columns
    assert pd.read_csv(out, encoding="utf-8").shape[0] == len(readings)
    assert list(predict_file(synthetic_bundle, xlsx_path)[PREDICTION_COLUMN]) == list(
        result[PREDICTION_COLUMN]
    )


def test_read_readings_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_readings(tmp_path / "none.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"N": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing feature columns"):
        read_readings(bad)


def test_predict_frame_does_not_mutate_input(
    synthetic_bundle: ModelBundle, synthetic_frame: pd.DataFrame
) -> None:
    readings = synthetic_frame.iloc[:3][list(FEATURES)].copy()
    predict_frame(synthetic_bundle, readings)
    assert PREDICTION_COLUMN not in readings.columns


def test_atomic_path_keeps_destination_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "keep.txt"
    target.write_text("original")
    with pytest.raises(RuntimeError), atomic_path(target) as tmp:
        tmp.write_text("partial")
        raise RuntimeError("boom")
    assert target.read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_write_json_and_timer(tmp_path: Path) -> None:
    path = write_json(tmp_path / "m.json", {"accuracy": 0.958, "name": "café"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"accuracy": 0.958, "name": "café"}
    with timer() as watch:
        sum(range(1000))
    assert watch.seconds >= 0
