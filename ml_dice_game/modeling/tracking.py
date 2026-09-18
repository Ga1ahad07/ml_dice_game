# ml_dice_game/modeling/tracking.py
from contextlib import contextmanager
import mlflow
from loguru import logger


class ExperimentTracker:
    """Envuelve el ciclo de vida de un run de MLflow."""

    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def run(self, run_name: str):
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"MLflow run iniciado: {run.info.run_id}")
            yield run

    @staticmethod
    def log_cv_results(cv_results: dict):
        for metric_name, values in cv_results.items():
            if metric_name.startswith("test_"):
                mlflow.log_metric(
                    f"cv_{metric_name.removeprefix('test_')}_mean",
                    float(values.mean()),
                )

    @staticmethod
    def log_params(params: dict):
        mlflow.log_params({key: str(value) for key, value in params.items()})

    @staticmethod
    def log_test_metrics(metrics: dict, prefix: str = ""):
        for k, v in metrics.items():
            mlflow.log_metric(f"{prefix}{k}", v)

    @staticmethod
    def log_model(model, model_name: str, registered_model_name: str | None = None):
        # XGBoost y RandomForest exponen distinta API de logging nativa de MLflow
        flavor = mlflow.xgboost if model.__class__.__module__.startswith(
            "xgboost") else mlflow.sklearn
        flavor.log_model(model, artifact_path="model",
                         registered_model_name=registered_model_name)
        logger.success(f"Modelo {model_name} logueado en MLflow" +
                       (f" y registrado como {registered_model_name}" if registered_model_name else ""))

    @staticmethod
    def log_figure_dir(figures_dir):
        mlflow.log_artifacts(str(figures_dir), artifact_path="figures")

    @staticmethod
    def promote_to_production(registered_model_name: str, version: int):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=registered_model_name, version=version, stage="Production",
            archive_existing_versions=True,
        )
        logger.success(
            f"{registered_model_name} v{version} promovido a Production")
