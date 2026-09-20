import json
from pathlib import Path

import joblib
import pandas as pd
import typer

from ml_dice_game.config import (
    FINAL_FEATURES_PATH,
    FINAL_MODEL_PATH,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    load_params,
)
from ml_dice_game.dataset import DiceGameDataset
from ml_dice_game.explain import ShapExplainer
from ml_dice_game.features import DataSplitter
from ml_dice_game.modeling.common import CrossValidatedTrainer, save_confusion_matrix, save_roc_curve
from ml_dice_game.modeling.evaluate_test import TestEvaluator
from ml_dice_game.modeling.select_best_model import BestModelSelector
from ml_dice_game.modeling.train_random_forest import RandomForestFactory
from ml_dice_game.modeling.train_xgboost import XGBoostFactory
from ml_dice_game.modeling.tracking import ExperimentTracker
from ml_dice_game.modeling.tune_best_model import ModelTuner

app = typer.Typer()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@app.command()
def validate(input_path: Path = RAW_DATA_DIR / "dataset2.csv") -> None:
    dataset = DiceGameDataset(input_path)
    dataset.load()
    report = dataset.validate_game_rules()
    summary = dataset.summary()
    _write_json(
        INTERIM_DATA_DIR / "validation_report.json",
        {
            "shape": list(dataset.df.shape),
            "n_missing": int(summary["n_missing"]),
            "n_duplicates": int(summary["n_duplicates"]),
            **report,
        },
    )


@app.command()
def split(
    input_path: Path = RAW_DATA_DIR / "dataset2.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test.csv",
) -> None:
    frame = pd.read_csv(input_path, sep=";")
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test, _ = splitter.split(frame)
    train = X_train.copy()
    train[TARGET_COLUMN] = y_train
    test = X_test.copy()
    test[TARGET_COLUMN] = y_test
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_output_path, index=False)
    test.to_csv(test_output_path, index=False)


def _load_training_data():
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    X = train.drop(columns=[TARGET_COLUMN])
    y = train[TARGET_COLUMN]
    params = load_params().get("split", {})
    splitter = DataSplitter(cv_splits=int(params.get("cv_n_splits", 5)))
    return X, y, splitter.get_cv()


@app.command()
def train() -> None:
    X, y, cv = _load_training_data()
    factories = {
        "RandomForest": RandomForestFactory.create(),
        "XGBoost": XGBoostFactory.create(),
    }
    results = {}
    tracker_config = load_params().get("mlflow", {})
    tracker = ExperimentTracker(tracker_config.get(
        "experiment_name", "ml_dice_game"))
    for name, estimator in factories.items():
        result = CrossValidatedTrainer(name, estimator, cv).fit(X, y)
        results[name] = result
        model_path = MODELS_DIR / "base" / f"{name.lower()}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.estimator, model_path)
        figure_dir = FIGURES_DIR / "oof"
        confusion_path = save_confusion_matrix(
            y, result.oof_predictions, figure_dir /
            f"{name.lower()}_confusion_matrix.png", f"{name}: OOF confusion matrix"
        )
        roc_path = save_roc_curve(
            y, result.oof_probabilities, figure_dir /
            f"{name.lower()}_roc_auc.png", f"{name}: OOF ROC-AUC"
        )
        with tracker.run(name, {"stage": "cv_oof", "model": name}):
            tracker.log_result(result.metrics, artifacts=[
                               confusion_path, roc_path])
            tracker.log_model(result.estimator, f"{name.lower()}_model")
    selected = BestModelSelector().select(results)
    _write_json(
        REPORTS_DIR / "metrics" / "cv.json",
        {name: result.metrics for name, result in results.items()},
    )
    _write_json(REPORTS_DIR / "metrics" /
                "selected_model.json", {"name": selected.name})


@app.command()
def tune() -> None:
    X, y, cv = _load_training_data()
    selected = json.loads(
        (REPORTS_DIR / "metrics" / "selected_model.json").read_text(encoding="utf-8"))["name"]
    estimator = {
        "RandomForest": RandomForestFactory.create(),
        "XGBoost": XGBoostFactory.create(),
    }[selected]
    search = ModelTuner.from_config(estimator, cv).fit(X, y)
    final_model = search.best_estimator_
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN]
    result = TestEvaluator(
        FIGURES_DIR / "test").evaluate(selected + "_tuned", final_model, X_test, y_test)
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, FINAL_MODEL_PATH)
    FINAL_FEATURES_PATH.write_text(json.dumps(
        list(X.columns), indent=2), encoding="utf-8")
    _write_json(REPORTS_DIR / "metrics" / "test.json", result["metrics"])
    tracker_config = load_params().get("mlflow", {})
    tracker = ExperimentTracker(tracker_config.get(
        "experiment_name", "ml_dice_game"))
    with tracker.run("final_tuned", {"stage": "test", "model": selected}):
        tracker.log_result(
            result["metrics"],
            params=search.best_params_,
            artifacts=list(result["artifacts"].values()),
        )
        tracker.log_model(final_model, "final_model")


@app.command()
def explain() -> None:
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    X_test = test.drop(columns=[TARGET_COLUMN])
    explainer = ShapExplainer(joblib.load(FINAL_MODEL_PATH), X_test)
    output_dir = FIGURES_DIR / "shap"
    paths = [
        output_dir / "global_importance.png",
        output_dir / "beeswarm.png",
    ]
    explainer.save_global_importance(paths[0])
    explainer.save_global_beeswarm(paths[1])
    tracker_config = load_params().get("mlflow", {})
    tracker = ExperimentTracker(tracker_config.get(
        "experiment_name", "ml_dice_game"))
    with tracker.run("final_shap", {"stage": "explain"}):
        tracker.log_result({}, artifacts=paths)


if __name__ == "__main__":
    app()
