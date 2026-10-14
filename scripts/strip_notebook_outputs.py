#!/usr/bin/env python3
"""Strip cell outputs and execution counts from Jupyter notebooks.

Only outputs, execution counts and widget state are removed; cell sources are left
untouched, with one exception: Colab ``# prompt: ...`` comment lines are dropped so
archived notebooks carry only the authors' own code.

Example:
    python scripts/strip_notebook_outputs.py old/*.ipynb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

LOGGER = logging.getLogger("strip_notebook_outputs")
PROMPT_MARKER = "# prompt:"


def strip_notebook(notebook: dict) -> dict:
    """Return ``notebook`` with outputs, counters and widget state removed."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        cell["source"] = [
            line for line in cell.get("source", []) if not line.lstrip().startswith(PROMPT_MARKER)
        ]
    notebook.get("metadata", {}).pop("widgets", None)
    return notebook


def process(path: Path, *, check: bool) -> bool:
    """Strip a single notebook. Return True if the file changed (or would change)."""
    with path.open(encoding="utf-8") as handle:
        original = handle.read()
    stripped = json.dumps(strip_notebook(json.loads(original)), indent=1, ensure_ascii=False) + "\n"
    if stripped == original:
        return False
    if not check:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(stripped)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("notebooks", nargs="+", type=Path, help="Notebook files to clean.")
    parser.add_argument(
        "--check", action="store_true", help="Do not write; exit 1 if any notebook would change."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    changed = [path for path in args.notebooks if process(path, check=args.check)]
    for path in changed:
        LOGGER.info("%s %s", "would strip" if args.check else "stripped", path)
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
