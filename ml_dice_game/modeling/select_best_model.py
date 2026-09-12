import json
from pathlib import Path

import pandas as pd
import typer

from ml_dice_game.config import MODELS_DIR, REPORTS_DIR
from ml_dice_game.plots import EvaluationReporter
from ml_dice_game.modeling.common import save_json


app = typer.Typer()


@app.command()
def main(
    rf_metrics_path: Path = REPORTS_DIR / "metrics" / "random_forest.json",
    xgb_metrics_path: Path = REPORTS_DIR / "metrics" / "xgboost.json",
    selection_path: Path = MODELS_DIR / "selection.json",
    output_metrics_path: Path = REPORTS_DIR / "metrics" / "base_models.json",
) -> None:
    rf_metrics = json.loads(rf_metrics_path.read_text(encoding="utf-8"))
    xgb_metrics = json.loads(xgb_metrics_path.read_text(encoding="utf-8"))
    metrics = {
        "RandomForest": {key: rf_metrics[key] for key in ("R2", "MAE", "RMSE")},
        "XGBoost": {key: xgb_metrics[key] for key in ("R2", "MAE", "RMSE")},
    }
    metrics_df = pd.DataFrame(metrics).T
    selected_model = metrics_df["R2"].idxmax()
    model_filename = "random_forest.pkl" if selected_model == "RandomForest" else "xgboost.pkl"

    save_json(
        {
            "selected_model": selected_model,
            "model_path": str(MODELS_DIR / "base" / model_filename),
            "selection_metric": "R2",
            "metrics": metrics,
        },
        selection_path,
    )
    save_json(metrics, output_metrics_path)
    EvaluationReporter(save=True).plot_metrics_comparison(
        metrics_df,
        filename="base_models_metrics_comparison.png",
    )


if __name__ == "__main__":
    app()
