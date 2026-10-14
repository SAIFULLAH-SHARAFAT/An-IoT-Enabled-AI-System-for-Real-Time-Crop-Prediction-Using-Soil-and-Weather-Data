"""LIME explanation for one test-set prediction."""

from __future__ import annotations

import argparse

from crop_recommendation.cli.common import (
    add_data_arguments,
    add_output_argument,
    existing_file,
    load_data,
)
from crop_recommendation.explain import DEFAULT_SAMPLES, explain_instance, save_explanation
from crop_recommendation.persistence import ModelBundle


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "explain", help="Explain a prediction with LIME.", description=__doc__
    )
    parser.add_argument("--model-path", type=existing_file, required=True)
    parser.add_argument(
        "--index", type=int, default=5, help="Row of the test split (default: %(default)s)."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="LIME neighbourhood size (default: %(default)s).",
    )
    add_data_arguments(parser)
    add_output_argument(parser, "reports/lime")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    try:
        bundle = ModelBundle.load(args.model_path)
        data = load_data(args)
        explanation = explain_instance(
            bundle, data, args.index, seed=args.seed, num_samples=args.n_samples
        )
    except ImportError as error:
        raise SystemExit(
            "LIME needs the extra: pip install 'crop-recommendation[explain]'"
        ) from error
    for path in save_explanation(explanation, args.output_dir, f"lime_instance_{args.index}"):
        print(path)  # noqa: T201 - primary command output
    return 0
