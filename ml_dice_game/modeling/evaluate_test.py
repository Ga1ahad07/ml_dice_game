from dataclasses import dataclass
from pathlib import Path

from ml_dice_game.modeling.common import (
    calculate_metrics,
    save_confusion_matrix,
    save_roc_curve,
)


@dataclass
class TestEvaluator:
    figures_dir: Path

    def evaluate(self, name, model, X_test, y_test) -> dict:
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, predictions, probabilities)
        prefix = self.figures_dir / name.lower().replace(" ", "_")
        artifacts = {
            "confusion_matrix": save_confusion_matrix(
                y_test, predictions, prefix.with_name(
                    prefix.name + "_confusion_matrix.png"), f"{name}: confusion matrix"
            ),
            "roc_curve": save_roc_curve(
                y_test, probabilities, prefix.with_name(
                    prefix.name + "_roc_auc.png"), f"{name}: ROC-AUC"
            ),
        }
        return {"metrics": metrics, "artifacts": artifacts, "predictions": predictions, "probabilities": probabilities}
