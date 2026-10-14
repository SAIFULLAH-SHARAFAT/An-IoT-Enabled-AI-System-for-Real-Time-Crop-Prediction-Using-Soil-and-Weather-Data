"""``crop-rec`` command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from crop_recommendation import __version__
from crop_recommendation.cli import edge, evaluate, explain, predict, reproduce, train
from crop_recommendation.logging_config import LOG_LEVELS, configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crop-rec",
        description="IoT-enabled AI system for real-time crop prediction (soil and weather data).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity (default: %(default)s).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for module in (train, evaluate, predict, explain, edge, reproduce):
        module.add_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
