"""Logging setup shared by every command-line entry point."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once; later calls only adjust the level."""
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S")
    logging.getLogger().setLevel(level.upper())
