import json

from ml_dice_game.modeling.select_best_model import load_model_metrics


def test_selection_metrics_are_cv_only(tmp_path):
    metrics_path = tmp_path / "random_forest.json"
    metrics_path.write_text(
        json.dumps(
            {
                "cv_ROC_AUC_mean": 0.91,
                "OOF_ROC_AUC": 0.90,
                "test_ROC_AUC": 0.10,
            }
        ),
        encoding="utf-8",
    )

    metrics = load_model_metrics(metrics_path)

    assert metrics["cv_ROC_AUC_mean"] == 0.91
    assert metrics["OOF_ROC_AUC"] == 0.90