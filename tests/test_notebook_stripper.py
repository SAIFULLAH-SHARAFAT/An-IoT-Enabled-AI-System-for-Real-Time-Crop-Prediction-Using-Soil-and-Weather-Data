from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "strip_notebook_outputs.py"


@pytest.fixture(scope="module")
def stripper():
    if not SCRIPT.is_file():
        pytest.skip("scripts/ is not part of this installation")
    spec = importlib.util.spec_from_file_location("strip_notebook_outputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook() -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title"]},
            {
                "cell_type": "code",
                "execution_count": 7,
                "outputs": [{"output_type": "stream", "text": ["42"]}],
                "source": ["# prompt: plot things\n", "x = 1\n", "# keep this comment\n"],
            },
        ],
        "metadata": {"widgets": {"big": "state"}, "kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_strip_removes_outputs_but_keeps_sources(stripper) -> None:
    cleaned = stripper.strip_notebook(notebook())
    code = cleaned["cells"][1]
    assert code["outputs"] == [] and code["execution_count"] is None
    assert code["source"] == ["x = 1\n", "# keep this comment\n"]
    assert cleaned["cells"][0]["source"] == ["# Title"]
    assert "widgets" not in cleaned["metadata"] and "kernelspec" in cleaned["metadata"]


def test_check_mode_does_not_write(stripper, tmp_path: Path) -> None:
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(notebook()), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    assert stripper.main(["--check", str(path)]) == 1
    assert path.read_text(encoding="utf-8") == before
    assert stripper.main([str(path)]) == 0
    assert stripper.main(["--check", str(path)]) == 0  # already clean
