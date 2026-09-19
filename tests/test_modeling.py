import json
import math

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml_dice_game.modeling.common import (
    evaluate_model,
    evaluate_predictions,
    save_json,
    split_xy,
)


def test_split_xy_removes_target():
    frame = pd.DataFrame({"RONDA": [1, 2], "VENTAJA": [0, 1]})
    features, target = split_xy(frame)
    assert list(features.columns) == ["RONDA"]
    assert target.tolist() == [0, 1]
    assert "VENTAJA" not in features.columns


def test_evaluate_model_returns_common_metric_names():
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    model = DummyClassifier(strategy="prior").fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert set(metrics) == {"Accuracy", "Precision", "Recall", "F1", "ROC_AUC"}
    assert all(math.isfinite(value) for value in metrics.values())


def test_save_json_creates_parent(tmp_path):
    path = tmp_path / "metrics" / "result.json"
    save_json({"ROC_AUC": 0.5}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["ROC_AUC"] == 0.5


def test_evaluate_predictions_returns_oof_metrics():
    X = pd.DataFrame({"feature": range(12)})
    y = pd.Series([0, 1] * 6)

    model = DummyClassifier(strategy="prior")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    predictions = cross_val_predict(model, X, y, cv=cv, method="predict")
    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    metrics = evaluate_predictions(y, predictions, probabilities)

    assert set(metrics) == {
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    }
    assert all(math.isfinite(value) for value in metrics.values())
