import json
from pathlib import Path

import pandas as pd
import typer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from ml_dice_game.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_params,
)
from ml_dice_game.explain import ShapExplainer
from ml_dice_game.modeling.common import (
    evaluate_model,
    load_model,
    load_train_test,
    make_cv,
    save_json,
    save_model,
    split_xy,
)
from ml_dice_game.plots import EvaluationReporter


app = typer.Typer()


@app.command()
def main(
    selection_path: Path = MODELS_DIR / "selection.json",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    metrics_path: Path = REPORTS_DIR / "metrics" / "base_vs_tuned.json",
) -> None:
    params = load_params()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_name = selection["selected_model"]

    base_path = Path(selection["model_path"])
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    if not base_path.is_file():
        raise FileNotFoundError(
            f"El modelo base seleccionado no existe: {base_path}. "
            "Ejecute primero los stages de entrenamiento y selección."
        )

    train_frame, test_frame = load_train_test(train_path, test_path)
    X_train, y_train = split_xy(train_frame)
    X_test, y_test = split_xy(test_frame)
    base_model = load_model(base_path)

    if selected_name == "XGBoost":
        estimator = XGBClassifier(
            random_state=params["random_state"],
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        params_key = "xgb_param_distributions"
    else:
        estimator = RandomForestClassifier(
            random_state=params["random_state"],
            n_jobs=-1,
        )
        params_key = "rf_param_distributions"

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params["tune"][params_key],
        n_iter=params["tune"]["n_iter"],
        scoring="roc_auc",
        cv=make_cv(
            params["split"]["cv_n_splits"],
            params["random_state"],
        ),
        random_state=params["random_state"],
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_

    base_metrics = evaluate_model(base_model, X_test, y_test)
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test)

    save_model(tuned_model, X_train.columns.tolist(), model_path)
    save_json(
        {
            "target": "VENTAJA",
            "selected_model": selected_name,
            "selection_metric": selection["selection_metric"],
            "base": base_metrics,
            "tuned": tuned_metrics,
            "best_params": search.best_params_,
        },
        metrics_path,
    )

    comparison = pd.DataFrame(
        {"base": base_metrics, "tuned": tuned_metrics}
    ).T
    EvaluationReporter(save=True).plot_metrics_comparison(
        comparison,
        filename="base_vs_tuned_metrics_comparison.png",
    )
    ShapExplainer(tuned_model, X_test).save_global_importance(
        FIGURES_DIR / "shap_global_importance.png"
    )


if __name__ == "__main__":
    app()
