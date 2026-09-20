from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_dvc_declares_reproducible_pipeline():
    pipeline = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    assert list(pipeline["stages"]) == ["validate",
                                        "split", "train", "tune", "explain"]
    assert "models/final/model.joblib" in pipeline["stages"]["tune"]["outs"]
    assert "reports/figures/shap" in pipeline["stages"]["explain"]["outs"]
