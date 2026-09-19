import json
from pathlib import Path

import pandas as pd
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from xgboost import XGBClassifier

from ml_dice_game.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_params,
)
from ml_dice_game.explain import ShapExplainer
from ml_dice_game.modeling.common import (
    evaluate_predictions,
    load_model,
    make_cv,
    save_json,
    save_model,
    split_xy,
)
from ml_dice_game.modeling.tracking import ExperimentTracker
from ml_dice_game.plots import EvaluationReporter


app = typer.Typer()


@app.command()
def main(
    selection_path: Path = MODELS_DIR / "selection.json",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
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
        raise FileNotFoundError(f"El modelo base no existe: {base_path}")

    train_frame = pd.read_csv(train_path)
    X_train, y_train = split_xy(train_frame)
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

    cv = make_cv(
        params["split"]["cv_n_splits"],
        params["random_state"],
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params["tune"][params_key],
        n_iter=params["tune"]["n_iter"],
        scoring=params["tune"]["scoring"],
        cv=cv,
        random_state=params["random_state"],
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_

    # Predicciones OOF para ambos modelos; no se utiliza test.
    base_oof_predictions = cross_val_predict(
        base_model,
        X_train,
        y_train,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )
    base_oof_probabilities = cross_val_predict(
        base_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    tuned_oof_predictions = cross_val_predict(
        tuned_model,
        X_train,
        y_train,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )
    tuned_oof_probabilities = cross_val_predict(
        tuned_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    base_metrics = evaluate_predictions(
        y_train,
        base_oof_predictions,
        base_oof_probabilities,
    )
    tuned_metrics = evaluate_predictions(
        y_train,
        tuned_oof_predictions,
        tuned_oof_probabilities,
    )

    save_model(tuned_model, X_train.columns.tolist(), model_path)

    save_json(
        {
            "target": "VENTAJA",
            "selected_model": selected_name,
            "selection_metric": selection["selection_metric"],
            "base_oof": base_metrics,
            "tuned_oof": tuned_metrics,
            "best_params": search.best_params_,
        },
        metrics_path,
    )

    comparison = pd.DataFrame(
        {"base": base_metrics, "tuned": tuned_metrics}
    ).T

    reporter = EvaluationReporter(save=True)

    for metric in ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC"):
        reporter.plot_metric_comparison(
            comparison,
            metric,
            f"base_vs_tuned_oof_{metric.lower()}_comparison.png",
        )

    reporter.plot_confusion_matrix_from_predictions(
        y_train,
        tuned_oof_predictions,
        "Tuned_OOF",
    )
    reporter.plot_roc_curve_from_predictions(
        y_train,
        tuned_oof_probabilities,
        "Tuned_OOF",
    )

    # SHAP utiliza únicamente datos de entrenamiento.
    tuned_model.fit(X_train, y_train)

    group_map = {
        **{
            column: "Carta"
            for column in X_train.columns
            if column.startswith("C")
        },
        **{
            column: "Tablero"
            for column in X_train.columns
            if column.startswith("T")
        },
        "RONDA": "Ronda",
        "TURNO": "Turno",
    }

    shap_explainer = ShapExplainer(tuned_model, X_train)
    shap_paths = {
        "global": FIGURES_DIR / "shap_global_importance.png",
        "group": FIGURES_DIR / "shap_group_importance.png",
        "beeswarm": FIGURES_DIR / "shap_summary_beeswarm.png",
    }

    shap_explainer.save_global_importance(shap_paths["global"])
    shap_explainer.save_group_importance(group_map, shap_paths["group"])
    shap_explainer.save_global_beeswarm(shap_paths["beeswarm"])

    mlflow_params = params.get("mlflow", {})
    tracker = ExperimentTracker(
        mlflow_params.get("experiment_name", "ml_dice_game")
    )
    with tracker.run("Tuned"):
        tracker.log_params(
            {
                "selected_model": selected_name,
                "scoring": params["tune"]["scoring"],
                "n_iter": params["tune"]["n_iter"],
                **search.best_params_,
            }
        )
        tracker.log_oof_metrics(tuned_metrics, prefix="tuned_oof_")
        tracker.log_model(
            tuned_model,
            "Tuned",
            registered_model_name=mlflow_params.get(
                "registered_model_name"
            ),
        )
        tracker.log_figure_dir(FIGURES_DIR)


if __name__ == "__main__":
    app()
