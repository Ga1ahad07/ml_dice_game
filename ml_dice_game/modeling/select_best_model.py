from dataclasses import dataclass

from ml_dice_game.modeling.common import TrainingResult


@dataclass
class BestModelSelector:
    """Selects only candidate models using validation ROC_AUC."""

    candidate_names: tuple[str, ...] = ("RandomForest", "XGBoost")

    def select(self, results: dict[str, TrainingResult]) -> TrainingResult:
        candidates = {name: results[name] for name in self.candidate_names}
        if not candidates:
            raise ValueError("No hay modelos candidatos para seleccionar")
        return max(candidates.values(), key=lambda result: result.metrics["ROC_AUC"])
