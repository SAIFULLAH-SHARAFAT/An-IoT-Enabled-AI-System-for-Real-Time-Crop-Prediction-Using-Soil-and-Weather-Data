"""Batch inference on raw (unscaled) sensor readings."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from crop_recommendation.config import FEATURES
from crop_recommendation.persistence import ModelBundle

LOGGER = logging.getLogger(__name__)

PREDICTION_COLUMN = "predicted_crop"


def read_readings(path: Path | str) -> pd.DataFrame:
    """Read readings from ``.csv`` or ``.xlsx`` and check the feature columns are present."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, engine="openpyxl")
    else:
        frame = pd.read_csv(path, encoding="utf-8")
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing feature columns: {missing}")
    return frame


def predict_frame(bundle: ModelBundle, readings: pd.DataFrame) -> pd.DataFrame:
    """Return ``readings`` with a ``predicted_crop`` column appended."""
    result: pd.DataFrame = readings.copy()
    result[PREDICTION_COLUMN] = bundle.predict(readings)
    return result


def predict_file(
    bundle: ModelBundle, input_path: Path | str, output_path: Path | str | None = None
) -> pd.DataFrame:
    """Predict every row of ``input_path``; optionally write the result to ``output_path``."""
    result = predict_frame(bundle, read_readings(input_path))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False, encoding="utf-8")
        LOGGER.info("Wrote %d predictions to %s", len(result), output_path)
    return result
