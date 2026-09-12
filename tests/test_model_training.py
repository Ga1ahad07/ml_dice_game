import pytest
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import json
import pandas as pd

from ml_dice_game.modeling.train_model import TrainModel
from ml_dice_game.modeling.train_random_forest import RandomForestTrainer
from ml_dice_game.modeling.train_xgboost import XGBoostTrainer


@pytest.fixture
def small_params():
    return {
        "random_state": 42,
        "train": {
            "rf": {"n_estimators": 3},
            "xgb": {"n_estimators": 3},
        },
        "split": {"cv_n_splits": 2},
    }


def test_both_trainers_inherit_from_train_model():
    assert issubclass(RandomForestTrainer, TrainModel)
    assert issubclass(XGBoostTrainer, TrainModel)


def test_random_forest_child_builds_expected_estimator(small_params):
    trainer = RandomForestTrainer(params=small_params)
    estimator = trainer.build_estimator()

    assert isinstance(estimator, RandomForestRegressor)
    assert estimator.n_estimators == 3
    assert trainer.model_name == "RandomForest"


def test_xgboost_child_builds_expected_estimator(small_params):
    trainer = XGBoostTrainer(params=small_params)
    estimator = trainer.build_estimator()

    assert isinstance(estimator, XGBRegressor)
    assert estimator.n_estimators == 3
    assert trainer.model_name == "XGBoost"


def test_model_params_include_reproducibility_options(small_params):
    trainer = RandomForestTrainer(params=small_params)

    assert trainer.model_params()["random_state"] == 42
    assert trainer.model_params()["n_jobs"] == -1


def test_run_writes_model_and_metrics(tmp_path, small_params):
    train = pd.DataFrame({
        "RONDA": [1, 1, 2, 2, 3, 3],
        "feature": [1, 2, 3, 4, 5, 6],
        "PUNTAJE": [2, 3, 4, 5, 6, 7],
    })
    test = pd.DataFrame({
        "RONDA": [1, 2],
        "feature": [7, 8],
        "PUNTAJE": [8, 9],
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "models" / "rf.pkl"
    metrics_path = tmp_path / "metrics" / "rf.json"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    trainer = RandomForestTrainer(
        train_path=train_path,
        test_path=test_path,
        model_path=model_path,
        metrics_path=metrics_path,
        params=small_params,
    )
    payload = trainer.run()

    assert model_path.exists()
    assert model_path.with_suffix(".features.json").exists()
    assert metrics_path.exists()
    assert payload["model"] == "RandomForest"
    assert set(("R2", "MAE", "RMSE")).issubset(payload)
    assert json.loads(metrics_path.read_text())["model"] == "RandomForest"
