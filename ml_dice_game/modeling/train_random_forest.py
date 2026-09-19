from pathlib import Path

import typer
from sklearn.ensemble import RandomForestClassifier

from ml_dice_game.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ml_dice_game.modeling.train_model import TrainModel


class RandomForestTrainer(TrainModel):
    @property
    def model_name(self) -> str:
        return "RandomForest"

    @property
    def train_params_key(self) -> str:
        return "rf"

    def build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.model_params())


app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    model_path: Path = MODELS_DIR / "base" / "random_forest.pkl",
    metrics_path: Path = REPORTS_DIR / "metrics" / "random_forest.json",
) -> None:
    trainer = RandomForestTrainer(
        train_path=train_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )
    trainer.run()


if __name__ == "__main__":
    app()
