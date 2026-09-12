import json

import pandas as pd
from sklearn.dummy import DummyRegressor

from ml_dice_game.modeling.common import evaluate_model, save_json, split_xy


def test_split_xy_removes_target():
    frame = pd.DataFrame({"RONDA": [1, 2], "PUNTAJE": [3, 4]})
    features, target = split_xy(frame)
    assert list(features.columns) == ["RONDA"]
    assert target.tolist() == [3, 4]


def test_evaluate_model_returns_common_metric_names():
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([1, 2, 3])
    model = DummyRegressor(strategy="mean").fit(X, y)
    assert set(evaluate_model(model, X, y)) == {"R2", "MAE", "RMSE"}


def test_save_json_creates_parent(tmp_path):
    path = tmp_path / "metrics" / "result.json"
    save_json({"R2": 0.5}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["R2"] == 0.5
