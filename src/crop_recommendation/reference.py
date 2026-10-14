"""Results published in the paper, used by ``crop-rec reproduce`` for side-by-side comparison.

Sharafat et al., "An IoT-enabled AI system for real-time crop prediction using soil and
weather data in precision agriculture", Smart Agricultural Technology 12 (2025) 101263.
Accuracies are fractions in [0, 1].
"""

from __future__ import annotations

from typing import Final

#: Table 4 - default classifiers, test accuracy.
TABLE4_DEFAULT_ACCURACY: Final[dict[str, float]] = {
    "nb": 0.926,
    "knn": 0.880,
    "dt": 0.930,
    "rf": 0.958,
    "lr": 0.836,
    "svc": 0.905,
    "adaboost": 0.100,
    "gb": 0.955,
    "catboost": 0.926,
}

#: Fig. 13 - accuracy after hyper-parameter tuning.
FIG13_TUNED_ACCURACY: Final[dict[str, float]] = {
    "dt": 0.938,
    "nb": 0.926,
    "lr": 0.911,
    "rf": 0.958,
    "svc": 0.897,
    "knn": 0.929,
    "adaboost": 0.926,
    "gb": 0.945,
    "catboost": 0.949,
}

#: Table 5 - stacking ensembles: (test accuracy, best cross-validation accuracy).
TABLE5_STACKING: Final[dict[str, tuple[float, float]]] = {
    "dt": (0.935, 0.880),
    "rf": (0.955, 0.909),
    "nb": (0.945, 0.925),
    "knn": (0.940, 0.920),
    "lr": (0.945, 0.899),
    "svc": (0.959, 0.948),
    "adaboost": (0.905, 0.890),
    "gb": (0.925, 0.898),
    "catboost": (0.932, 0.931),
}

#: Table 6 - deep models: (accuracy, trainable parameters).
TABLE6_DEEP: Final[dict[str, tuple[float, int]]] = {
    "tabnet": (0.920, 444400),
    "cnn": (0.896, 21206),
    "lstm": (0.918, 117334),
    "cnn_lstm": (0.889, 22582),
    "cnn_bilstm": (0.868, 530050),
}

#: Inference times reported for Random Forest and Gradient Boosting on the Raspberry Pi 5 (seconds).
PI5_INFERENCE_SECONDS: Final[dict[str, float]] = {"rf": 0.0608, "gb": 0.0693}
