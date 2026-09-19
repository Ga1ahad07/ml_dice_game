import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from ml_dice_game.config import MODELS_DIR, REPORTS_DIR, load_params
from ml_dice_game.modeling.common import save_json
from ml_dice_game.plots import EvaluationReporter


app = typer.Typer()


def load_model_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@app.command()
def main(
    rf_metrics_path: Path = REPORTS_DIR / "metrics" / "random_forest.json",
    xgb_metrics_path: Path = REPORTS_DIR / "metrics" / "xgboost.json",
    selection_path: Path = MODELS_DIR / "selection.json",
    output_metrics_path: Path = REPORTS_DIR / "metrics" / "base_models.json",
) -> None:
    rf_metrics = load_model_metrics(rf_metrics_path)
    xgb_metrics = load_model_metrics(xgb_metrics_path)

    metrics = {
        "RandomForest": {
            "CV_Accuracy": rf_metrics["cv_Accuracy_mean"],
            "CV_Precision": rf_metrics["cv_Precision_mean"],
            "CV_Recall": rf_metrics["cv_Recall_mean"],
            "CV_F1": rf_metrics["cv_F1_mean"],
            "CV_ROC_AUC": rf_metrics["cv_ROC_AUC_mean"],
            "OOF_Accuracy": rf_metrics["OOF_Accuracy"],
            "OOF_Precision": rf_metrics["OOF_Precision"],
            "OOF_Recall": rf_metrics["OOF_Recall"],
            "OOF_F1": rf_metrics["OOF_F1"],
            "OOF_ROC_AUC": rf_metrics["OOF_ROC_AUC"],
        },
        "XGBoost": {
            "CV_Accuracy": xgb_metrics["cv_Accuracy_mean"],
            "CV_Precision": xgb_metrics["cv_Precision_mean"],
            "CV_Recall": xgb_metrics["cv_Recall_mean"],
            "CV_F1": xgb_metrics["cv_F1_mean"],
            "CV_ROC_AUC": xgb_metrics["cv_ROC_AUC_mean"],
            "OOF_Accuracy": xgb_metrics["OOF_Accuracy"],
            "OOF_Precision": xgb_metrics["OOF_Precision"],
            "OOF_Recall": xgb_metrics["OOF_Recall"],
            "OOF_F1": xgb_metrics["OOF_F1"],
            "OOF_ROC_AUC": xgb_metrics["OOF_ROC_AUC"],
        },
    }

    metrics_df = pd.DataFrame(metrics).T

    params = load_params()
    selection_metric = (
        f"CV_{params.get('selection', {}).get('metric', 'roc_auc').upper()}"
    )

    selected_model = metrics_df[selection_metric].idxmax()

    model_filename = (
        "random_forest.pkl"
        if selected_model == "RandomForest"
        else "xgboost.pkl"
    )

    save_json(
        {
            "target": "VENTAJA",
            "selected_model": selected_model,
            "model_path": str(MODELS_DIR / "base" / model_filename),
            "selection_metric": selection_metric,
            "selection_value": float(metrics_df.loc[selected_model, selection_metric]),
            "metrics": metrics,
        },
        selection_path,
    )

    save_json(metrics, output_metrics_path)

    # La selección y la gráfica comparativa usan exclusivamente métricas CV.
    cv_metrics_df = metrics_df.rename(
        columns={
            "CV_Accuracy": "Accuracy",
            "CV_Precision": "Precision",
            "CV_Recall": "Recall",
            "CV_F1": "F1",
            "CV_ROC_AUC": "ROC_AUC",
        }
    )

    EvaluationReporter(save=True).plot_metrics_comparison(
        cv_metrics_df,
        filename="base_models_metrics_comparison.png",
    )


if __name__ == "__main__":
    app()
