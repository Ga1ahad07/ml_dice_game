from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

from ml_dice_game.config import BOARD_COLUMNS, CARD_COLUMNS, FIGURES_DIR, TARGET_COLUMN


class _FigureReporter:
    def __init__(self, save_dir: Path = FIGURES_DIR, save: bool = False):
        self.save_dir = save_dir
        self.save = save
        sns.set_theme(style="whitegrid")

    def _finish(self, fig, filename: str):
        fig.tight_layout()
        if self.save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.save_dir / filename
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Figura guardada en {output_path}")
        plt.show()
        return fig


class EDAReporter(_FigureReporter):
    """Genera las figuras EDA de las secciones 1.3 a 1.7 del notebook."""

    def __init__(self, df: pd.DataFrame, save_dir: Path = FIGURES_DIR, save: bool = False):
        super().__init__(save_dir=save_dir, save=save)
        self.df = df

    def plot_ronda_turno_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, column, color in zip(
            axes,
            ["RONDA", "TURNO"],
            ["steelblue", "darkorange"],
        ):
            counts = self.df[column].value_counts().sort_index()
            sns.barplot(x=counts.index, y=counts.values, ax=ax, color=color)
            ax.set_title(f"Distribucion de {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Frecuencia")
            for index, value in enumerate(counts.values):
                ax.text(index, value, str(value), ha="center", va="bottom")
        return self._finish(fig, "ronda_turno_distribution.png")

    def plot_target_distribution(self):
        counts = self.df[TARGET_COLUMN].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(x=counts.index.astype(str),
                    y=counts.values, ax=ax, color="seagreen")
        ax.set_title(f"Distribucion de {TARGET_COLUMN}")
        ax.set_xlabel(TARGET_COLUMN)
        ax.set_ylabel("Frecuencia")
        for index, value in enumerate(counts.values):
            ax.text(index, value, str(value), ha="center", va="bottom")
        return self._finish(fig, "target_distribution.png")

    def plot_card_positions(self):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        for ax, column in zip(axes.ravel(), CARD_COLUMNS):
            self.df[column].value_counts().sort_index().plot(
                kind="bar",
                ax=ax,
                color="slateblue",
            )
            ax.set_title(column)
            ax.set_xlabel("Valor")
            ax.set_ylabel("Frecuencia")
        return self._finish(fig, "card_positions.png")

    def plot_board_correlation_heatmap(self):
        correlation = self.df[BOARD_COLUMNS].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Multicolinealidad entre variables del tablero")
        self._finish(fig, "board_correlation_heatmap.png")
        return correlation

    def plot_correlation_heatmap(self, feature_cols: list[str]):
        """Conserva la interfaz anterior para el pipeline existente."""
        correlation = self.df[feature_cols].corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(correlation, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de correlacion entre variables predictoras")
        self._finish(fig, "correlation_heatmap.png")
        return correlation

    def plot_target_correlation(self, feature_cols: list[str]):
        target_corr = (
            self.df[feature_cols + [TARGET_COLUMN]]
            .corr(numeric_only=True)[TARGET_COLUMN]
            .drop(TARGET_COLUMN)
            .sort_values(key=abs, ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        target_corr.plot(kind="barh", color="teal", ax=ax)
        ax.set_title(f"Correlacion lineal con {TARGET_COLUMN}")
        ax.set_xlabel("Correlacion de Pearson")
        ax.invert_yaxis()
        self._finish(fig, "target_correlation.png")
        return target_corr

    def plot_group_target_correlation(self, target_corr: pd.Series, group_map: dict[str, str]):
        grouped = (
            target_corr.abs()
            .groupby(target_corr.index.map(group_map))
            .sum()
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped.plot(kind="barh", color="steelblue", ax=ax)
        ax.set_title(f"Correlacion absoluta por grupo con {TARGET_COLUMN}")
        ax.set_xlabel("Sumatoria de |correlacion|")
        self._finish(fig, "group_target_correlation.png")
        return grouped


class EvaluationReporter(_FigureReporter):
    """Genera las figuras de evaluacion de clasificacion del notebook."""

    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        filename: str = "metrics_comparison.png",
    ) -> None:
        metric_columns = [
            column
            for column in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
            if column in metrics_df.columns
        ]
        if not metric_columns:
            raise ValueError(
                "metrics_df no contiene metricas de clasificacion")

        fig, axes = plt.subplots(
            1,
            len(metric_columns),
            figsize=(4 * len(metric_columns), 4),
            squeeze=False,
        )
        for ax, metric in zip(axes.ravel(), metric_columns):
            metrics_df[metric].plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title(metric)
            ax.set_ylim(0, 1)
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=30)
        self._finish(fig, filename)

    def plot_metric_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric: str,
        filename: str,
    ) -> None:
        if metric not in metrics_df.columns:
            raise ValueError(f"metrics_df no contiene la metrica {metric}")

        fig, ax = plt.subplots(figsize=(5, 4))
        metrics_df[metric].plot(kind="bar", ax=ax, color=[
                                "steelblue", "darkorange"])
        ax.set_title(f"Comparacion base vs. tuneado: {metric}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=0)
        self._finish(fig, filename)

    def plot_confusion_matrix(self, model, X_test, y_test, model_name: str):
        predictions = model.predict(X_test)
        matrix = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["0", "1"],
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{model_name}: matriz de confusion")
        ax.grid(False)
        self._finish(fig, f"confusion_matrix_{model_name}.png")
        return matrix

    def plot_roc_curves(
        self,
        fitted_models: dict,
        X_test,
        y_test,
        filename: str = "roc_curves.png",
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
        for name, model in fitted_models.items():
            probabilities = model.predict_proba(X_test)[:, 1]
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test,
                probabilities,
            )
            auc = roc_auc_score(y_test, probabilities)
            ax.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{name} (AUC = {auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.5)")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Curva ROC - comparacion de modelos")
        ax.legend()
        self._finish(fig, filename)

    def plot_classification_report(self, model, X_test, y_test, model_name: str):
        from sklearn.metrics import classification_report

        report = classification_report(
            y_test,
            model.predict(X_test),
            target_names=["VENTAJA=0", "VENTAJA=1"],
            output_dict=True,
        )
        report_df = pd.DataFrame(report).T
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(report_df.iloc[:, :3], annot=True,
                    fmt=".3f", cmap="Blues", ax=ax)
        ax.set_title(f"Reporte de clasificacion - {model_name}")
        self._finish(fig, f"classification_report_{model_name}.png")
        return report_df

    def plot_feature_importances(self, model, feature_cols: list[str], title: str):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        fig, ax = plt.subplots(figsize=(7, 6))
        importances.sort_values(ascending=True).plot(kind="barh", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Importancia")
        self._finish(fig, f"feature_importances_{title.replace(' ', '_')}.png")
        return importances
