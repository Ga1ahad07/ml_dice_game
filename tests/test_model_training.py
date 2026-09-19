import json

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml_dice_game.modeling.train_model import TrainModel
from ml_dice_game.modeling.train_random_forest import RandomForestTrainer
from ml_dice_game.modeling.train_xgboost import XGBoostTrainer


@pytest.fixture
def small_params():
    return {
        "random_state": 42,
        "train": {
            "rf": {"n_estimators": 3},
            "xgb": {
                "n_estimators": 3,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
            },
        },
        "split": {"cv_n_splits": 2},
    }


def test_both_trainers_inherit_from_train_model():
    assert issubclass(RandomForestTrainer, TrainModel)
    assert issubclass(XGBoostTrainer, TrainModel)


def test_random_forest_child_builds_expected_estimator(small_params):
    trainer = RandomForestTrainer(params=small_params)
    estimator = trainer.build_estimator()

    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.n_estimators == 3
    assert trainer.model_name == "RandomForest"


def test_xgboost_child_builds_expected_estimator(small_params):
    trainer = XGBoostTrainer(params=small_params)
    estimator = trainer.build_estimator()

    assert isinstance(estimator, XGBClassifier)
    assert estimator.n_estimators == 3
    assert trainer.model_name == "XGBoost"


def test_model_params_include_reproducibility_options(small_params):
    trainer = RandomForestTrainer(params=small_params)

    assert trainer.model_params()["random_state"] == 42
    assert trainer.model_params()["n_jobs"] == -1


def test_run_writes_model_and_metrics(tmp_path, small_params):
    train = pd.DataFrame({
        "RONDA": [1, 1, 2, 2, 3, 3, 4, 4],
        "feature": [1, 2, 3, 4, 5, 6, 7, 8],
        "VENTAJA": [0, 1, 0, 1, 0, 1, 0, 1],
    })
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "models" / "rf.pkl"
    metrics_path = tmp_path / "metrics" / "rf.json"
    train.to_csv(train_path, index=False)

    trainer = RandomForestTrainer(
        train_path=train_path,
        model_path=model_path,
        metrics_path=metrics_path,
        params=small_params,
    )
    payload = trainer.run()

    assert model_path.exists()
    assert model_path.with_suffix(".features.json").exists()
    assert metrics_path.exists()
    assert payload["model"] == "RandomForest"
    assert payload["target"] == "VENTAJA"
    assert set(("OOF_Accuracy", "OOF_Precision", "OOF_Recall",
               "OOF_F1", "OOF_ROC_AUC")).issubset(payload)
    assert set((
        "cv_Accuracy_mean",
        "cv_Precision_mean",
        "cv_Recall_mean",
        "cv_F1_mean",
        "cv_ROC_AUC_mean",
    )).issubset(payload)
    assert json.loads(metrics_path.read_text())["model"] == "RandomForest"


def test_run_writes_model_and_oof_metrics(tmp_path, small_params):
    train = pd.DataFrame({
        "RONDA": [1, 1, 2, 2, 3, 3, 4, 4],
        "feature": [1, 2, 3, 4, 5, 6, 7, 8],
        "VENTAJA": [0, 1, 0, 1, 0, 1, 0, 1],
    })

    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "models" / "rf.pkl"
    metrics_path = tmp_path / "metrics" / "rf.json"
    train.to_csv(train_path, index=False)

    trainer = RandomForestTrainer(
        train_path=train_path,
        model_path=model_path,
        metrics_path=metrics_path,
        params=small_params,
    )

    payload = trainer.run()

    assert model_path.exists()
    assert model_path.with_suffix(".features.json").exists()
    assert metrics_path.exists()

    assert payload["model"] == "RandomForest"
    assert payload["target"] == "VENTAJA"

    assert set((
        "OOF_Accuracy",
        "OOF_Precision",
        "OOF_Recall",
        "OOF_F1",
        "OOF_ROC_AUC",
    )).issubset(payload)

    assert set((
        "cv_Accuracy_mean",
        "cv_Precision_mean",
        "cv_Recall_mean",
        "cv_F1_mean",
        "cv_ROC_AUC_mean",
    )).issubset(payload)

    saved_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved_payload["model"] == "RandomForest"
    assert "test" not in json.dumps(saved_payload).lower()
