from pathlib import Path
import typer
from loguru import logger
from pathlib import Path
import joblib
import pandas as pd
import json
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from xgboost import XGBRegressor

from ml_dice_game.config import (
    RANDOM_STATE, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TARGET_COLUMN, load_params,
)
from ml_dice_game.dataset import DiceGameDataset
from ml_dice_game.features import DataSplitter
from ml_dice_game.plots import EvaluationReporter
from ml_dice_game.modeling.tracking import ExperimentTracker


class ModelTrainer:
    """Entrena los modelos base con validación cruzada (sección 3 del notebook)."""

    SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error"}

    def __init__(self, cv, random_state: int = RANDOM_STATE):
        self.cv = cv
        self.random_state = random_state
        self.models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ),
            "XGBoost": XGBRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        }
        self.cv_results_: dict = {}
        self.fitted_models_: dict = {}

    def cross_validate_all(self, X_train, y_train) -> dict:
        for name, model in self.models.items():
            res = cross_validate(model, X_train, y_train,
                                 cv=self.cv, scoring=self.SCORING, n_jobs=-1)
            self.cv_results_[name] = res
            logger.info(
                f"{name} CV -> R2: {res['test_r2'].mean():.4f} | "
                f"MAE: {-res['test_mae'].mean():.4f} | RMSE: {-res['test_rmse'].mean():.4f}"
            )
        return self.cv_results_

    def fit_all(self, X_train, y_train) -> dict:
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.fitted_models_[name] = model
        return self.fitted_models_

    def save(self, model, feature_cols, path=MODELS_DIR / "model.pkl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.success(f"Modelo guardado en {path}")
        json.dump(feature_cols, open(path.with_suffix(".features.json"), "w"))


class ModelEvaluator:
    """Evalúa modelos entrenados sobre test y elige el ganador (sección 4)."""

    @staticmethod
    def evaluate(model, X_test, y_test) -> dict:
        preds = model.predict(X_test)
        return {
            "R2": r2_score(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        }

    def evaluate_all(self, fitted_models: dict, X_test, y_test) -> pd.DataFrame:
        results = {name: self.evaluate(model, X_test, y_test)
                   for name, model in fitted_models.items()}
        return pd.DataFrame(results).T

    @staticmethod
    def pick_best(metrics_df: pd.DataFrame, by: str = "R2") -> str:
        best = metrics_df[by].idxmax()
        logger.info(f"Modelo ganador según {by}: {best}")
        return best


class HyperparameterTuner:
    """RandomizedSearchCV sobre el modelo ganador (sección 5)."""

    def __init__(self, estimator, param_distributions: dict, cv, n_iter: int = 50,
                 scoring: str = "r2", random_state: int = RANDOM_STATE):
        self.search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )

    def tune(self, X_train, y_train):
        self.search.fit(X_train, y_train)
        logger.success(f"Mejor R2 en CV: {self.search.best_score_:.4f}")
        logger.info(f"Mejores hiperparámetros: {self.search.best_params_}")
        return self.search.best_estimator_


app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    params = load_params()
    tracker = ExperimentTracker(
        experiment_name=params["mlflow"]["experiment_name"])

    dataset_train = DiceGameDataset(path=train_path, sep=",").load()
    dataset_test = DiceGameDataset(path=test_path, sep=",").load()
    X_train = dataset_train.drop(columns=[TARGET_COLUMN])
    y_train = dataset_train[TARGET_COLUMN]
    X_test = dataset_test.drop(columns=[TARGET_COLUMN])
    y_test = dataset_test[TARGET_COLUMN]

    splitter = DataSplitter(stratify_col=params["split"]["stratify_col"])
    cv = splitter.get_cv()

    with tracker.run(run_name="train_base_models"):
        mlflow.log_params(params["train"])
        trainer = ModelTrainer(cv=cv, random_state=params["random_state"])
        cv_results = trainer.cross_validate_all(X_train, y_train)
        tracker.log_cv_results(cv_results)
        fitted_models = trainer.fit_all(X_train, y_train)

        evaluator = ModelEvaluator()
        metrics_df = evaluator.evaluate_all(fitted_models, X_test, y_test)
        best_name = evaluator.pick_best(metrics_df)
        tracker.log_test_metrics(
            metrics_df.loc[best_name].to_dict(), prefix="base_")

    with tracker.run(run_name=f"tune_{best_name}"):
        param_key = "xgb_param_distributions" if best_name == "XGBoost" else "rf_param_distributions"
        mlflow.log_params(
            {"n_iter": params["tune"]["n_iter"], "model": best_name})

        tuner = HyperparameterTuner(
            fitted_models[best_name].__class__(
                random_state=params["random_state"], n_jobs=-1),
            params["tune"][param_key], cv=cv, n_iter=params["tune"]["n_iter"],
        )
        tuned_model = tuner.tune(X_train, y_train)

        tuned_metrics = evaluator.evaluate(tuned_model, X_test, y_test)
        tracker.log_test_metrics(tuned_metrics, prefix="tuned_")

        reporter = EvaluationReporter(save_dir=FIGURES_DIR, save=True)
        reporter.plot_metrics_comparison(metrics_df)
        tracker.log_figure_dir(FIGURES_DIR)

        tracker.log_model(
            tuned_model, model_name=best_name,
            registered_model_name=params["mlflow"]["registered_model_name"],
        )

    # también en disco, para DVC/FastAPI fallback
    trainer.save(
        tuned_model, feature_cols=X_train.columns.tolist(), path=model_path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(pd.Series(tuned_metrics).to_json())
    logger.success("Entrenamiento + tracking completo.")


if __name__ == "__main__":
    app()
