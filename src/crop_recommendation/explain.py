"""LIME explanations of single predictions (paper Section 3.4, Fig. 24).

Explanations are computed in the Min-Max scaled feature space, the space the model was
trained in, so the reported thresholds read like ``0.26 < Rainfall(cm) <= 0.41``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from crop_recommendation.data import PreparedData
from crop_recommendation.persistence import ModelBundle
from crop_recommendation.utils import atomic_path

if TYPE_CHECKING:
    from lime.explanation import Explanation

LOGGER = logging.getLogger(__name__)

DEFAULT_SAMPLES = 5000  # LIME's own default neighbourhood size


def build_explainer(bundle: ModelBundle, background: pd.DataFrame, seed: int):  # type: ignore[no-untyped-def]
    """LIME tabular explainer fitted on the (scaled) training data."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        background.to_numpy(),
        feature_names=list(background.columns),
        class_names=[str(name) for name in bundle.label_encoder.classes_],
        mode="classification",
        random_state=seed,
    )


def explain_instance(
    bundle: ModelBundle,
    data: PreparedData,
    index: int,
    *,
    seed: int,
    num_features: int | None = None,
    num_samples: int = DEFAULT_SAMPLES,
) -> Explanation:
    """Explain the prediction for row ``index`` of the held-out test set."""
    if not hasattr(bundle.model, "predict_proba"):
        raise TypeError(
            f"{type(bundle.model).__name__} does not provide predict_proba; LIME needs it"
        )
    if not 0 <= index < len(data.X_test):
        raise IndexError(f"index must be in [0, {len(data.X_test) - 1}], got {index}")
    explainer = build_explainer(bundle, data.X_train, seed)
    row = data.X_test.iloc[index]
    return explainer.explain_instance(
        row.to_numpy(),
        lambda values: bundle.model.predict_proba(
            pd.DataFrame(values, columns=data.X_test.columns)
        ),
        num_features=num_features or data.X_test.shape[1],
        num_samples=num_samples,
        top_labels=1,
    )


def save_explanation(explanation: Explanation, output_dir: Path | str, stem: str) -> list[Path]:
    """Write the interactive HTML report and a static PNG bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    html_path = output_dir / f"{stem}.html"
    png_path = output_dir / f"{stem}.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    with atomic_path(html_path) as tmp:
        explanation.save_to_file(str(tmp))

    label = explanation.available_labels()[0]
    figure = explanation.as_pyplot_figure(label=label)
    figure.tight_layout()
    with atomic_path(png_path) as tmp:
        figure.savefig(tmp, dpi=300, format="png")
    plt.close(figure)
    LOGGER.info("Saved LIME explanation to %s and %s", html_path, png_path)
    return [html_path, png_path]
