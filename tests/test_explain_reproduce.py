from __future__ import annotations

from pathlib import Path

import pytest

from conftest import make_frame
from crop_recommendation.cli.reproduce import Row, render_markdown
from crop_recommendation.data import PreparedData, prepare
from crop_recommendation.persistence import ModelBundle


def test_row_delta_and_markdown() -> None:
    rows = [
        Row("Table 4 (default)", "Random Forest", 0.958, 0.9576),
        Row("Table 6 (deep)", "TabNet", 0.920, 0.9364, "444400 params"),
    ]
    assert rows[1].delta_points == pytest.approx(1.64)
    markdown = render_markdown(rows)
    assert markdown.splitlines()[0].startswith("| Section | Item | Paper | Reproduced")
    assert "| Random Forest | 95.8 | 95.8 | -0.0 |" in markdown
    assert "444400 params" in markdown


def test_lime_explanation_is_saved(
    tmp_path: Path, synthetic_bundle: ModelBundle, synthetic_data: PreparedData
) -> None:
    pytest.importorskip("lime")
    from crop_recommendation.explain import explain_instance, save_explanation

    explanation = explain_instance(synthetic_bundle, synthetic_data, 0, seed=42, num_samples=200)
    paths = save_explanation(explanation, tmp_path / "lime", "instance_0")
    assert {p.suffix for p in paths} == {".html", ".png"}
    assert all(p.stat().st_size > 0 for p in paths)


def test_lime_input_validation(synthetic_bundle: ModelBundle, synthetic_data: PreparedData) -> None:
    pytest.importorskip("lime")
    from crop_recommendation.explain import explain_instance

    with pytest.raises(IndexError):
        explain_instance(synthetic_bundle, synthetic_data, 10_000, seed=42)
    no_proba = ModelBundle(
        model=object(),  # type: ignore[arg-type]
        scaler=synthetic_bundle.scaler,
        label_encoder=synthetic_bundle.label_encoder,
    )
    with pytest.raises(TypeError, match="predict_proba"):
        explain_instance(no_proba, synthetic_data, 0, seed=42)


@pytest.mark.deep
@pytest.mark.slow
def test_tabnet_fits_and_reports_parameters() -> None:
    pytest.importorskip("pytorch_tabnet")
    from crop_recommendation.models.deep.tabnet import fit_tabnet

    # TabNet's paper configuration (batch 256) needs a few hundred updates before it starts learning.
    data = prepare(make_frame(per_class=700))
    result = fit_tabnet(data, seed=42)
    assert result.test_metrics.accuracy > 0.8
    assert result.n_parameters > 0
    assert result.epochs_run >= 1
    assert "loss" in result.to_dict()["history"]
