"""Recommend crops for raw sensor readings in a CSV or XLSX file."""

from __future__ import annotations

import argparse
from pathlib import Path

from crop_recommendation.cli.common import existing_file
from crop_recommendation.config import FEATURES
from crop_recommendation.inference import predict_file
from crop_recommendation.persistence import ModelBundle


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "predict",
        help="Predict crops for a table of readings.",
        description=f"{__doc__}\n\nRequired columns: {', '.join(FEATURES)}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=existing_file, required=True)
    parser.add_argument(
        "--input", type=existing_file, required=True, help="CSV or XLSX of raw readings."
    )
    parser.add_argument("--output", type=Path, default=None, help="Write predictions to this CSV.")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    bundle = ModelBundle.load(args.model_path)
    result = predict_file(bundle, args.input, args.output)
    if args.output is None:
        print(result.to_string(index=False))  # noqa: T201 - primary command output
    return 0
