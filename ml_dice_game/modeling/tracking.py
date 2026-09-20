from contextlib import contextmanager
from pathlib import Path
import mlflow
import xgboost
import mlflow.xgboost
import mlflow.sklearn


class ExperimentTracker:
    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):

        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None):
        with self.mlflow.start_run(run_name=run_name, tags=tags or {}):
            yield self

    def log_result(self, metrics: dict[str, float], params: dict | None = None, artifacts=None):
        if metrics:
            self.mlflow.log_metrics(
                {key: float(metrics[key]) for key in metrics})
        if params:
            self.mlflow.log_params({key: str(value)
                                   for key, value in params.items()})
        for artifact in artifacts or []:
            path = Path(artifact)
            if path.exists():
                self.mlflow.log_artifact(str(path))

    def log_model(self, model, artifact_path: str) -> None:
        if isinstance(model, (xgboost.XGBClassifier, xgboost.XGBRegressor)):
            mlflow.xgboost.log_model(model, artifact_path)
        else:
            mlflow.sklearn.log_model(model, artifact_path)
