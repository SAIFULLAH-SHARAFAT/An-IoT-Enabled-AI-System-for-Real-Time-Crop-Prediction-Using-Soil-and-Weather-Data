"""Dataset loading, validation and preprocessing (paper Section 3.1-3.2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from crop_recommendation.config import (
    EXPECTED_CLASSES,
    EXPECTED_ROWS,
    FEATURES,
    SEED,
    TARGET,
    TEST_SIZE,
)

LOGGER = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the input table does not match the expected schema."""


@dataclass(frozen=True)
class PreparedData:
    """Scaled, label-encoded and split dataset together with the fitted transformers.

    The scaler and label encoder are fitted exactly as in the paper's experiments and are
    the objects deployed on the edge device.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    scaler: MinMaxScaler
    label_encoder: LabelEncoder

    @property
    def classes(self) -> NDArray[np.str_]:
        """Crop names indexed by encoded label."""
        return np.asarray(self.label_encoder.classes_)

    @property
    def class_names(self) -> list[str]:
        return [str(name) for name in self.classes]

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Read the crop workbook and keep only the feature and target columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_excel(path, engine="openpyxl")
    return validate_schema(frame)


def validate_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and numeric dtypes; return the columns in canonical order."""
    required = [*FEATURES, TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    frame = frame[required]
    non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric feature columns: {non_numeric}")
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop null rows and duplicate instances (Section 3.2)."""
    cleaned: pd.DataFrame = frame.dropna().drop_duplicates().reset_index(drop=True)
    removed = len(frame) - len(cleaned)
    if removed:
        LOGGER.warning("Removed %d null/duplicate rows", removed)
    if len(cleaned) != EXPECTED_ROWS or cleaned[TARGET].nunique() != EXPECTED_CLASSES:
        LOGGER.warning(
            "Dataset has %d rows and %d classes; the paper reports %d rows and %d classes",
            len(cleaned),
            cleaned[TARGET].nunique(),
            EXPECTED_ROWS,
            EXPECTED_CLASSES,
        )
    return cleaned


def prepare(
    frame: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> PreparedData:
    """Min-Max scale, label-encode and split the data with a stratified hold-out.

    Scaling is fitted on the full table before splitting, which reproduces the paper's
    preprocessing order and yields the same scaler that is shipped to the Raspberry Pi.
    """
    frame = clean(validate_schema(frame))

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(frame[list(FEATURES)]), columns=list(FEATURES))

    label_encoder = LabelEncoder()
    labels = np.asarray(label_encoder.fit_transform(frame[TARGET]), dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        scaled, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        label_encoder=label_encoder,
    )


def load_prepared(
    path: Path | str, *, test_size: float = TEST_SIZE, seed: int = SEED
) -> PreparedData:
    """Convenience wrapper: workbook path -> :class:`PreparedData`."""
    return prepare(load_dataset(path), test_size=test_size, seed=seed)
