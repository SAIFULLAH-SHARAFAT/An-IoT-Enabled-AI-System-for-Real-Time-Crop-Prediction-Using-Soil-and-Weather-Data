"""Argument helpers shared by the sub-commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from crop_recommendation.config import SEED, default_dataset_path
from crop_recommendation.data import PreparedData, load_prepared


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        metavar="XLSX",
        help="Crop workbook (default: $CROP_REC_DATASET, else the file under <repo>/data/).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed (default: %(default)s)."
    )


def add_output_argument(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(default),
        metavar="DIR",
        help="Where to write artifacts (default: %(default)s).",
    )


def existing_file(value: str) -> Path:
    """argparse type: a path that must point at an existing file."""
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


def load_data(args: argparse.Namespace) -> PreparedData:
    """Load and split the dataset selected on the command line."""
    path = args.data if args.data is not None else default_dataset_path()
    return load_prepared(path, seed=args.seed)
