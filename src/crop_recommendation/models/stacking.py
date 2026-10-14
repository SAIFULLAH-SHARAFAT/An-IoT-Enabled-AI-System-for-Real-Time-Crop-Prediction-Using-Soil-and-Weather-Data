"""Stacking ensembles (paper Algorithm 1 and Table 5).

Base learners feed their predictions to a meta-learner; the meta-learner's hyper-parameters
are tuned with a small grid / randomized search over the whole stack. The best configuration
reported in the paper uses Random Forest, Gradient Boosting and CatBoost as base learners and
an SVC meta-learner (95.9 % test accuracy).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BaseFactory = Callable[[int], BaseEstimator]
MetaFactory = Callable[[int], BaseEstimator]

#: Base learners as configured in the stacking experiments.
BASE_LEARNERS: dict[str, BaseFactory] = {
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=seed
    ),
    "gb": lambda seed: GradientBoostingClassifier(
        learning_rate=0.1, max_depth=5, n_estimators=150, subsample=0.8, random_state=seed
    ),
    "catboost": lambda seed: CatBoostClassifier(
        depth=4, iterations=1000, l2_leaf_reg=1, learning_rate=0.05, random_state=seed, verbose=0, allow_writing_files=False
    ),
    "dt": lambda seed: DecisionTreeClassifier(
        max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=seed
    ),
}


@dataclass(frozen=True)
class StackSpec:
    """One row of Table 5: which base learners, which meta-learner, how it is tuned."""

    key: str
    label: str
    base_keys: tuple[str, ...]
    meta: MetaFactory
    grid: Mapping[str, Sequence[Any]]
    cv: int
    n_iter: int | None = None  # set -> RandomizedSearchCV, otherwise GridSearchCV
    #: Base learners configured differently from ``BASE_LEARNERS`` in this experiment.
    base_overrides: Mapping[str, BaseFactory] = field(default_factory=dict)
    #: Base learners tuned (randomized search, 3 draws, 2-fold CV) before stacking.
    base_search: Mapping[str, Mapping[str, Sequence[Any]]] = field(default_factory=dict)


_DEFAULT_BASE = ("rf", "gb", "catboost")

STACK_SPECS: dict[str, StackSpec] = {
    spec.key: spec
    for spec in (
        StackSpec(
            "dt",
            "Decision Tree",
            _DEFAULT_BASE,
            lambda seed: DecisionTreeClassifier(random_state=seed),
            {
                "final_estimator__criterion": ["gini", "entropy"],
                "final_estimator__max_depth": [5, 10],
                "final_estimator__min_samples_split": [2],
                "final_estimator__min_samples_leaf": [1],
            },
            cv=2,
            n_iter=3,
            base_overrides={
                "gb": lambda seed: GradientBoostingClassifier(
                    learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8, random_state=seed
                ),
                "catboost": lambda seed: CatBoostClassifier(
                    depth=4,
                    iterations=500,
                    l2_leaf_reg=1,
                    learning_rate=0.05,
                    random_state=seed,
                    verbose=0, allow_writing_files=False,
                ),
            },
        ),
        StackSpec(
            "rf",
            "Random Forest",
            ("gb", "catboost", "dt"),
            lambda seed: RandomForestClassifier(random_state=seed),
            {"final_estimator__n_estimators": [100, 200], "final_estimator__max_depth": [10, 15]},
            cv=2,
            n_iter=3,
            base_overrides={
                "gb": lambda seed: GradientBoostingClassifier(random_state=seed, subsample=0.9),
                "catboost": lambda seed: CatBoostClassifier(
                    random_state=seed, verbose=0, allow_writing_files=False, iterations=500
                ),
                "dt": lambda seed: DecisionTreeClassifier(random_state=seed),
            },
            base_search={
                "gb": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 150],
                    "max_depth": [3, 5],
                },
                "catboost": {"learning_rate": [0.05, 0.1], "depth": [4, 5]},
                "dt": {
                    "max_depth": [5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
            },
        ),
        StackSpec(
            "nb",
            "GaussianNB",
            _DEFAULT_BASE,
            lambda seed: GaussianNB(),
            {"final_estimator__var_smoothing": [1e-9, 1e-8, 1e-7]},
            cv=3,
        ),
        StackSpec(
            "knn",
            "KNN",
            _DEFAULT_BASE,
            lambda seed: KNeighborsClassifier(),
            {
                "final_estimator__n_neighbors": [3, 5],
                "final_estimator__weights": ["uniform"],
                "final_estimator__metric": ["euclidean"],
            },
            cv=3,
        ),
        StackSpec(
            "lr",
            "Logistic Regression",
            _DEFAULT_BASE,
            # liblinear is one-vs-rest by nature; recent scikit-learn requires that to be explicit.
            lambda seed: OneVsRestClassifier(LogisticRegression(max_iter=1000, solver="liblinear")),
            {"final_estimator__estimator__C": [0.1, 1.0]},
            cv=2,
        ),
        StackSpec(
            "svc",
            "SVC",
            _DEFAULT_BASE,
            lambda seed: SVC(probability=True, random_state=seed),
            {"final_estimator__C": [0.1, 1.0], "final_estimator__kernel": ["linear", "rbf"]},
            cv=2,
        ),
        StackSpec(
            "adaboost",
            "AdaBoost",
            _DEFAULT_BASE,
            lambda seed: AdaBoostClassifier(random_state=seed),
            {
                "final_estimator__n_estimators": [50, 100],
                "final_estimator__learning_rate": [0.01, 0.1],
            },
            cv=5,
            base_overrides={
                "rf": lambda seed: RandomForestClassifier(
                    bootstrap=False,
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=seed,
                ),
                "gb": lambda seed: GradientBoostingClassifier(
                    subsample=0.8,
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=seed,
                ),
            },
        ),
        StackSpec(
            "gb",
            "Gradient Boosting",
            ("rf", "dt", "catboost"),
            lambda seed: GradientBoostingClassifier(random_state=seed),
            {
                "final_estimator__n_estimators": [50, 100],
                "final_estimator__learning_rate": [0.01, 0.1],
                "final_estimator__max_depth": [3, 4],
            },
            cv=2,
        ),
        StackSpec(
            "catboost",
            "CatBoost",
            ("rf", "dt", "gb"),
            lambda seed: CatBoostClassifier(
                iterations=50, learning_rate=0.1, depth=3, random_state=seed, verbose=0, allow_writing_files=False
            ),
            {
                "final_estimator__iterations": [50],
                "final_estimator__learning_rate": [0.01, 0.1],
                "final_estimator__depth": [3],
            },
            cv=2,
            base_overrides={
                "rf": lambda seed: RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=seed
                ),
                "gb": lambda seed: GradientBoostingClassifier(
                    subsample=0.8,
                    n_estimators=50,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=seed,
                ),
            },
        ),
    )
}

#: The configuration reported as best in the paper.
BEST_META = "svc"


def get_stack_spec(key: str) -> StackSpec:
    try:
        return STACK_SPECS[key]
    except KeyError:
        raise KeyError(
            f"Unknown meta-learner '{key}'. Available: {', '.join(STACK_SPECS)}"
        ) from None


def make_base(spec: StackSpec, key: str, seed: int) -> BaseEstimator:
    """Base learner ``key`` as configured for this stacking experiment."""
    return spec.base_overrides.get(key, BASE_LEARNERS[key])(seed)


def tune_bases(
    spec: StackSpec, seed: int, X: Any, y: Any, *, n_jobs: int | None = -1
) -> list[tuple[str, BaseEstimator]]:
    """Base learners of ``spec``; those listed in ``spec.base_search`` are tuned first."""
    estimators: list[tuple[str, BaseEstimator]] = []
    for key in spec.base_keys:
        estimator = make_base(spec, key, seed)
        if key in spec.base_search:
            search = RandomizedSearchCV(
                estimator,
                dict(spec.base_search[key]),
                n_iter=3,
                cv=2,
                random_state=seed,
                n_jobs=n_jobs,
            )
            search.fit(X, y)
            estimator = search.best_estimator_
        estimators.append((f"{key}_base", estimator))
    return estimators


def build_stack(
    spec: StackSpec, seed: int, estimators: Sequence[tuple[str, BaseEstimator]] | None = None
) -> StackingClassifier:
    """Assemble the (untuned) stacking classifier for ``spec``."""
    if estimators is None:
        estimators = [(f"{key}_base", make_base(spec, key, seed)) for key in spec.base_keys]
    return StackingClassifier(estimators=list(estimators), final_estimator=spec.meta(seed))


def build_stack_search(
    spec: StackSpec,
    seed: int,
    *,
    n_jobs: int | None = -1,
    estimators: Sequence[tuple[str, BaseEstimator]] | None = None,
) -> GridSearchCV | RandomizedSearchCV:
    """Hyper-parameter search over the meta-learner, as in Algorithm 1 (steps 5-6)."""
    stack = build_stack(spec, seed, estimators)
    common: dict[str, Any] = {"cv": spec.cv, "scoring": "accuracy", "n_jobs": n_jobs}
    if spec.n_iter is not None:
        return RandomizedSearchCV(stack, spec.grid, n_iter=spec.n_iter, random_state=seed, **common)
    return GridSearchCV(stack, spec.grid, **common)
