"""Classical machine-learning classifiers (paper Section 3.3, Tables 4 and Fig. 13).

Every ``default`` factory mirrors the "default classifier" rows of Table 4; every ``tuned``
factory carries the best hyper-parameters found by the grid searches in the original
experiments, and ``grid`` holds the search space that produced them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

Factory = Callable[[int], BaseEstimator]


@dataclass(frozen=True)
class ModelSpec:
    """A classifier family: how to build it untuned, tuned, and how to search it."""

    key: str
    label: str
    default: Factory
    tuned: Factory
    grid: Mapping[str, Sequence[Any]]
    cv_splits: int = 5
    shuffled_cv: bool = False
    n_iter: int | None = None  # set -> RandomizedSearchCV, otherwise GridSearchCV
    search_base: Factory | None = None  # estimator searched over, when it differs from `default`


def _catboost(seed: int, **params: Any) -> CatBoostClassifier:
    return CatBoostClassifier(random_state=seed, verbose=0, allow_writing_files=False, **params)


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="dt",
            label="Decision Tree",
            default=lambda seed: DecisionTreeClassifier(random_state=seed),
            tuned=lambda seed: DecisionTreeClassifier(
                criterion="entropy",
                max_depth=None,
                max_features=None,
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=seed,
            ),
            grid={
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [None, "sqrt", "log2"],
            },
        ),
        ModelSpec(
            key="rf",
            label="Random Forest",
            default=lambda seed: RandomForestClassifier(random_state=seed),
            tuned=lambda seed: RandomForestClassifier(
                max_depth=None,
                max_features="sqrt",
                min_samples_leaf=1,
                min_samples_split=2,
                n_estimators=200,
                random_state=seed,
            ),
            grid={
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        ),
        ModelSpec(
            key="nb",
            label="GaussianNB",
            default=lambda seed: GaussianNB(),
            tuned=lambda seed: GaussianNB(var_smoothing=1e-9),
            grid={"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        ),
        ModelSpec(
            key="lr",
            label="Logistic Regression",
            default=lambda seed: LogisticRegression(random_state=seed),
            tuned=lambda seed: LogisticRegression(
                C=100, max_iter=5000, solver="lbfgs", random_state=seed
            ),
            grid={
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear", "saga"],
            },
            shuffled_cv=True,
            search_base=lambda seed: LogisticRegression(random_state=seed, max_iter=500),
        ),
        ModelSpec(
            key="svc",
            label="SVC",
            default=lambda seed: SVC(random_state=seed),
            tuned=lambda seed: SVC(C=10, degree=3, gamma="scale", kernel="poly", random_state=seed),
            grid={
                "C": [0.01, 0.1, 1, 10, 100],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto", 0.1, 1],
            },
            shuffled_cv=True,
        ),
        ModelSpec(
            key="knn",
            label="KNN",
            default=lambda seed: KNeighborsClassifier(),
            tuned=lambda seed: KNeighborsClassifier(
                n_neighbors=11, weights="distance", metric="manhattan"
            ),
            grid={
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
            shuffled_cv=True,
        ),
        ModelSpec(
            key="adaboost",
            label="AdaBoost",
            default=lambda seed: AdaBoostClassifier(random_state=seed),
            tuned=lambda seed: AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                learning_rate=1,
                n_estimators=300,
                random_state=seed,
            ),
            grid={
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.5, 1],
                "estimator": [
                    DecisionTreeClassifier(max_depth=1),
                    DecisionTreeClassifier(max_depth=3),
                ],
            },
        ),
        ModelSpec(
            key="gb",
            label="Gradient Boosting",
            default=lambda seed: GradientBoostingClassifier(random_state=seed),
            tuned=lambda seed: GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=seed
            ),
            grid={
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
            shuffled_cv=True,
        ),
        ModelSpec(
            key="xgb",
            label="XGBoost",
            default=lambda seed: XGBClassifier(random_state=seed),
            tuned=lambda seed: XGBClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=5,
                reg_lambda=15,
                random_state=seed,
            ),
            grid={},
        ),
        ModelSpec(
            key="catboost",
            label="CatBoost",
            default=lambda seed: _catboost(
                seed, iterations=100, learning_rate=0.1, depth=4, loss_function="MultiClass"
            ),
            tuned=lambda seed: _catboost(
                seed,
                depth=4,
                iterations=500,
                l2_leaf_reg=5,
                learning_rate=0.1,
                loss_function="MultiClass",
            ),
            grid={
                "iterations": [300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6],
                "l2_leaf_reg": [1, 3, 5],
            },
            cv_splits=2,
            n_iter=10,
        ),
    )
}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError:
        raise KeyError(f"Unknown model '{key}'. Available: {', '.join(MODEL_SPECS)}") from None


def build_search(
    spec: ModelSpec, seed: int, *, n_jobs: int | None = -1
) -> GridSearchCV | RandomizedSearchCV:
    """Hyper-parameter search object reproducing the original tuning protocol for ``spec``."""
    if not spec.grid:
        raise ValueError(f"No search space is defined for '{spec.key}'")
    cv: int | KFold = (
        KFold(n_splits=spec.cv_splits, shuffle=True, random_state=seed)
        if spec.shuffled_cv
        else spec.cv_splits
    )
    common: dict[str, Any] = {"scoring": "accuracy", "cv": cv, "n_jobs": n_jobs}
    estimator = (spec.search_base or spec.default)(seed)
    if spec.n_iter is not None:
        return RandomizedSearchCV(
            estimator, spec.grid, n_iter=spec.n_iter, random_state=seed, **common
        )
    return GridSearchCV(estimator, spec.grid, **common)
