from pathlib import Path

import typer
from xgboost import XGBClassifier

from ml_dice_game.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ml_dice_game.modeling.train_model import TrainModel


class XGBoostTrainer(TrainModel):
    @property
    def model_name(self) -> str:
        return "XGBoost"

    @property
    def train_params_key(self) -> str:
        return "xgb"

    def build_estimator(self) -> XGBClassifier:
        return XGBClassifier(**self.model_params())


app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    model_path: Path = MODELS_DIR / "base" / "xgboost.pkl",
    metrics_path: Path = REPORTS_DIR / "metrics" / "xgboost.json",
) -> None:
    trainer = XGBoostTrainer(
        train_path=train_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )
    trainer.run()


if __name__ == "__main__":
    app()
