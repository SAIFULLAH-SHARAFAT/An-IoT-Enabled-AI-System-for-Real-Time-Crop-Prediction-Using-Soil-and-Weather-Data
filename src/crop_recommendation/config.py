"""Project-wide constants taken from the paper's experimental protocol (Section 3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

#: The eight model inputs, in the column order used for training and for the edge device.
FEATURES: Final[tuple[str, ...]] = (
    "N",
    "P",
    "K",
    "pH",
    "Temp(°C)",
    "Humidity(%)",
    "Moisture(%)",
    "Rainfall(cm)",
)
TARGET: Final[str] = "Crop"

#: Dataset facts reported in Section 3.1 (22 crops, 150 samples each, 3,300 rows).
EXPECTED_ROWS: Final[int] = 3300
EXPECTED_CLASSES: Final[int] = 22

#: Section 3.2: 80:20 stratified split. Seed 42 is used by every notebook.
SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.20

DATASET_FILENAME: Final[str] = "New Final DataSet Without Sulphur.xlsx"
DATASET_ENV_VAR: Final[str] = "CROP_REC_DATASET"

#: Artifact names written next to each other by ``crop-rec train``.
MODEL_BUNDLE_NAME: Final[str] = "model.joblib"
METRICS_NAME: Final[str] = "metrics.json"


def project_root() -> Path:
    """Repository root when running from a source checkout (``src`` layout)."""
    return Path(__file__).resolve().parents[2]


def default_dataset_path() -> Path:
    """Dataset location: ``$CROP_REC_DATASET`` if set, else ``<repo>/data/<DATASET_FILENAME>``."""
    override = os.environ.get(DATASET_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return project_root() / "data" / DATASET_FILENAME
