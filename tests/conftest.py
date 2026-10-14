"""Shared fixtures: a small synthetic crop table and the real dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crop_recommendation.config import FEATURES, SEED, TARGET, default_dataset_path
from crop_recommendation.data import PreparedData, prepare
from crop_recommendation.models.classical import get_spec
from crop_recommendation.persistence import ModelBundle

CROPS = ("Aman Rice", "Bean", "Tomato")


def make_frame(
    per_class: int = 40, crops: tuple[str, ...] = CROPS, seed: int = SEED
) -> pd.DataFrame:
    """Well-separated synthetic readings with the real schema."""
    rng = np.random.default_rng(seed)
    frames = []
    for index, crop in enumerate(crops):
        centre = np.array(
            [
                100 + 80 * index,
                30 + 20 * index,
                40 + 30 * index,
                5.0 + 0.5 * index,
                20 + 4 * index,
                60 + 5 * index,
                50 + 10 * index,
                70 + 40 * index,
            ],
            dtype=float,
        )
        values = centre + rng.normal(0, 2.0, size=(per_class, len(FEATURES)))
        frame = pd.DataFrame(values, columns=list(FEATURES))
        frame[TARGET] = crop
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture(scope="session")
def synthetic_data(synthetic_frame: pd.DataFrame) -> PreparedData:
    return prepare(synthetic_frame)


@pytest.fixture(scope="session")
def synthetic_bundle(synthetic_data: PreparedData) -> ModelBundle:
    model = get_spec("rf").tuned(SEED)
    model.fit(synthetic_data.X_train, synthetic_data.y_train)
    return ModelBundle(
        model=model, scaler=synthetic_data.scaler, label_encoder=synthetic_data.label_encoder
    )


@pytest.fixture(scope="session")
def dataset_path() -> Path:
    path = default_dataset_path()
    if not path.is_file():
        pytest.skip(f"dataset not available at {path}")
    return path
