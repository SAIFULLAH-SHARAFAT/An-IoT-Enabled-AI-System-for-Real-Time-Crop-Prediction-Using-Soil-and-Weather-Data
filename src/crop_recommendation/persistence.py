"""Save and load the trained model together with the transformers it depends on.

Bundles are ``joblib`` (pickle) files. Only load bundles from sources you trust: unpickling
can execute arbitrary code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from crop_recommendation.config import FEATURES
from crop_recommendation.utils import atomic_path

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Everything needed to turn raw sensor readings into a crop name."""

    model: BaseEstimator
    scaler: MinMaxScaler
    label_encoder: LabelEncoder
    feature_names: tuple[str, ...] = FEATURES
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, readings: pd.DataFrame) -> np.ndarray:
        """Predict crop names for raw (unscaled) readings with the canonical feature columns."""
        missing = [name for name in self.feature_names if name not in readings.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        scaled: Any = self.scaler.transform(readings[list(self.feature_names)])
        if hasattr(self.model, "feature_names_in_"):  # fitted on a DataFrame: keep the names
            scaled = pd.DataFrame(scaled, columns=list(self.feature_names))
        encoded = np.asarray(self.model.predict(scaled)).astype(int).ravel()
        return np.asarray(self.label_encoder.inverse_transform(encoded))

    def save(self, path: Path | str) -> Path:
        """Write the bundle atomically and return its path."""
        path = Path(path)
        with atomic_path(path) as tmp:
            joblib.dump(self, tmp)
        LOGGER.info("Saved model bundle to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> ModelBundle:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model bundle not found: {path}")
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(
                f"{path} does not contain a ModelBundle (found {type(bundle).__name__})"
            )
        return bundle

    @classmethod
    def from_legacy_files(
        cls, *, model_path: Path | str, scaler_path: Path | str, encoder_path: Path | str
    ) -> ModelBundle:
        """Load the three separate ``.pkl`` files used by the original Raspberry Pi script."""
        for candidate in (model_path, scaler_path, encoder_path):
            if not Path(candidate).is_file():
                raise FileNotFoundError(f"File not found: {candidate}")
        return cls(
            model=joblib.load(model_path),
            scaler=joblib.load(scaler_path),
            label_encoder=joblib.load(encoder_path),
        )


def write_json(path: Path | str, payload: Any) -> Path:
    """Write ``payload`` as pretty-printed UTF-8 JSON (atomically)."""
    path = Path(path)
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
