from types import SimpleNamespace

import numpy as np

from ml_dice_game.modeling.common import METRIC_NAMES, calculate_metrics
from ml_dice_game.modeling.select_best_model import BestModelSelector


def test_metrics_use_one_public_naming_convention():
    metrics = calculate_metrics(
        [0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.2, 0.4])
    assert tuple(metrics) == METRIC_NAMES
    assert all(isinstance(value, float) for value in metrics.values())


def test_selector_uses_roc_auc_only_for_candidates():
    results = {
        "RandomForest": SimpleNamespace(name="RandomForest", metrics={"ROC_AUC": 0.80}),
        "XGBoost": SimpleNamespace(name="XGBoost", metrics={"ROC_AUC": 0.87}),
    }
    selected = BestModelSelector().select(results)
    assert selected.name == "XGBoost"
