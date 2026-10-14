from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crop_recommendation import __version__
from crop_recommendation.cli.main import build_parser, main
from crop_recommendation.config import FEATURES, MODEL_BUNDLE_NAME
from crop_recommendation.persistence import ModelBundle


@pytest.fixture
def bundle_path(tmp_path: Path, synthetic_bundle: ModelBundle) -> Path:
    return synthetic_bundle.save(tmp_path / MODEL_BUNDLE_NAME)


def test_help_and_version_exit_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["--help"], ["train", "--help"], ["edge", "--help"], ["--version"]):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_subcommand_is_required() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_train_argument_validation() -> None:
    parser = build_parser()
    args = parser.parse_args(["train", "classical", "--model", "rf", "--variant", "tuned"])
    assert (args.family, args.model, args.seed, str(args.output_dir)) == (
        "classical",
        "rf",
        42,
        "models",
    )
    assert parser.parse_args(["train", "stack"]).meta == "svc"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "classical", "--model", "unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "--model-path", "does-not-exist.joblib", "--input", "x.csv"])


def test_train_classical_writes_bundle_and_metrics(
    tmp_path: Path, synthetic_frame: pd.DataFrame
) -> None:
    data_path = tmp_path / "crops.xlsx"
    synthetic_frame.to_excel(data_path, index=False)
    out = tmp_path / "models"
    code = main(
        [
            "--log-level",
            "ERROR",
            "train",
            "classical",
            "--model",
            "dt",
            "--data",
            str(data_path),
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "dt-tuned" / MODEL_BUNDLE_NAME).is_file()
    assert (out / "dt-tuned" / "metrics.json").is_file()


def test_evaluate_command(
    tmp_path: Path,
    bundle_path: Path,
    synthetic_frame: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_path = tmp_path / "crops.xlsx"
    synthetic_frame.to_excel(data_path, index=False)
    out = tmp_path / "eval"
    code = main(
        [
            "--log-level",
            "ERROR",
            "evaluate",
            "--model-path",
            str(bundle_path),
            "--data",
            str(data_path),
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert "acc=1.000" in capsys.readouterr().out
    assert {p.name for p in out.iterdir()} == {
        "metrics.json",
        "confusion_matrix.png",
        "roc_curves.png",
    }


def test_predict_command(
    tmp_path: Path,
    bundle_path: Path,
    synthetic_frame: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readings = tmp_path / "readings.csv"
    synthetic_frame.iloc[[0]][list(FEATURES)].to_csv(readings, index=False, encoding="utf-8")
    assert (
        main(
            [
                "--log-level",
                "ERROR",
                "predict",
                "--model-path",
                str(bundle_path),
                "--input",
                str(readings),
            ]
        )
        == 0
    )
    assert "Aman Rice" in capsys.readouterr().out


def test_edge_dry_run_needs_no_hardware_or_secrets(
    bundle_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WEATHER_API_KEY", raising=False)
    monkeypatch.delenv("TB_ACCESS_TOKEN", raising=False)
    csv_log = tmp_path / "edge.csv"
    code = main(
        [
            "--log-level",
            "ERROR",
            "edge",
            "--model-path",
            str(bundle_path),
            "--dry-run",
            "--startup-delay",
            "0",
            "--csv-log",
            str(csv_log),
        ]
    )
    assert code == 0
    assert csv_log.read_text(encoding="utf-8").count("\n") == 2


def test_edge_without_secrets_fails_fast(
    bundle_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WEATHER_API_KEY", raising=False)
    monkeypatch.delenv("TB_ACCESS_TOKEN", raising=False)
    assert main(["--log-level", "ERROR", "edge", "--model-path", str(bundle_path)]) == 2


def test_edge_missing_model_file_is_reported(tmp_path: Path) -> None:
    code = main(
        ["--log-level", "ERROR", "edge", "--model-path", str(tmp_path / "none.joblib"), "--dry-run"]
    )
    assert code == 2
