"""Score a saved model on the held-out 20 % test split."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from crop_recommendation.cli.common import (
    add_data_arguments,
    add_output_argument,
    existing_file,
    load_data,
)
from crop_recommendation.evaluation import evaluate, plot_confusion_matrix, plot_roc_curves
from crop_recommendation.persistence import ModelBundle, write_json

LOGGER = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate a saved model.", description=__doc__)
    parser.add_argument(
        "--model-path", type=existing_file, required=True, help="Bundle from `crop-rec train`."
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip the confusion-matrix and ROC figures."
    )
    add_data_arguments(parser)
    add_output_argument(parser, "reports/evaluation")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    bundle = ModelBundle.load(args.model_path)
    data = load_data(args)
    metrics = evaluate(bundle.model, data.X_test, data.y_test)
    print(metrics.summary())  # noqa: T201 - primary command output
    write_json(args.output_dir / "metrics.json", metrics.to_dict())

    if not args.no_plots:
        predictions = np.asarray(bundle.model.predict(data.X_test)).ravel()
        plot_confusion_matrix(
            data.y_test, predictions, data.class_names, args.output_dir / "confusion_matrix.png"
        )
        if hasattr(bundle.model, "predict_proba"):
            plot_roc_curves(
                data.y_test,
                bundle.model.predict_proba(data.X_test),
                data.class_names,
                args.output_dir / "roc_curves.png",
            )
    return 0
