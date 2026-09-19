from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

from ml_dice_game.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from ml_dice_game.modeling.common import evaluate_model, load_model, save_json, split_xy


app = typer.Typer()


@app.command()
def main(
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    metrics_path: Path = REPORTS_DIR / "metrics" / "test.json",
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """Evalúa exclusivamente el modelo ganador sobre el conjunto de test."""
    if not test_path.is_file():
        raise FileNotFoundError(f"No existe el conjunto de test: {test_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo ganador: {model_path}")

    test_frame = pd.read_csv(test_path)
    X_test, y_test = split_xy(test_frame)
    model = load_model(model_path)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(model, X_test, y_test)

    payload = {
        "target": TARGET_COLUMN,
        "model_path": str(model_path),
        "metrics": {f"TEST_{key}": value for key, value in metrics.items()},
    }
    save_json(payload, metrics_path)

    figures_dir.mkdir(parents=True, exist_ok=True)

    confusion_path = figures_dir / "test_confusion_matrix.png"
    matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["0", "1"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Modelo ganador: matriz de confusión en test")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    roc_path = figures_dir / "test_roc_curve.png"
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities,
    )
    auc = roc_auc_score(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Modelo ganador (AUC = {auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.5)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC-AUC en test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
