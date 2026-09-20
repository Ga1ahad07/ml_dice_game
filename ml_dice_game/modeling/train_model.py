from dataclasses import dataclass

import pandas as pd

from ml_dice_game.modeling.common import CrossValidatedTrainer, TrainingResult


@dataclass
class ModelTrainer:
    """Facade for training one candidate with the shared CV contract."""

    name: str
    estimator: object
    cv: object

    def train(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        return CrossValidatedTrainer(self.name, self.estimator, self.cv).fit(X, y)
