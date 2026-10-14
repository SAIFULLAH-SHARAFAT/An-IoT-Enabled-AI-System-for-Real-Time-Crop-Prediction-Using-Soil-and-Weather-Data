"""Small helpers: deterministic seeding, timing and atomic file writes."""

from __future__ import annotations

import os
import random
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (when installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Stopwatch:
    """Elapsed wall-clock seconds, populated when the :func:`timer` block exits."""

    seconds: float = 0.0


@contextmanager
def timer() -> Iterator[Stopwatch]:
    """Measure the wall-clock time of the enclosed block.

    >>> with timer() as watch:
    ...     pass
    >>> watch.seconds >= 0
    True
    """
    watch = Stopwatch()
    start = time.perf_counter()
    try:
        yield watch
    finally:
        watch.seconds = time.perf_counter() - start


@contextmanager
def atomic_path(destination: Path) -> Iterator[Path]:
    """Yield a temporary path next to ``destination`` and move it into place on success.

    Readers never observe a half-written file, and a failed write leaves any existing
    destination untouched.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        yield tmp
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)
