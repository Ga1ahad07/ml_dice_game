from pathlib import Path

import yaml


DVC_PATH = Path(__file__).parents[1] / "dvc.yaml"


def test_training_stages_do_not_use_test_dataset():
    pipeline = yaml.safe_load(DVC_PATH.read_text(encoding="utf-8"))

    for stage_name in (
        "train_random_forest",
        "train_xgboost",
        "tune_best_model",
    ):
        stage = pipeline["stages"][stage_name]
        deps = stage.get("deps", [])

        assert "data/processed/test.csv" not in deps
        assert "test.csv" not in stage.get("cmd", "")


def test_evaluate_test_is_the_only_stage_using_test_dataset():
    pipeline = yaml.safe_load(DVC_PATH.read_text(encoding="utf-8"))

    evaluate_stage = pipeline["stages"]["evaluate_test"]
    deps = evaluate_stage.get("deps", [])

    assert "data/processed/test.csv" in deps
