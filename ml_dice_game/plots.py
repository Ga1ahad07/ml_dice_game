import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from ml_dice_game.config import FIGURES_DIR, CARD_COLUMNS, TARGET_COLUMN


class EDAReporter:
    """Genera y opcionalmente guarda las figuras exploratorias (sección 1 del notebook)."""

    def __init__(self, df: pd.DataFrame, save_dir=FIGURES_DIR, save: bool = False):
        self.df = df
        self.save_dir = save_dir
        self.save = save
        sns.set_theme(style="whitegrid")

    def _finish(self, fig, filename: str):
        fig.tight_layout()
        if self.save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_dir / filename, dpi=150)
            logger.info(f"Figura guardada en {self.save_dir / filename}")
        plt.show()

    def plot_ronda_turno_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self.df["RONDA"].value_counts().sort_index().plot(
            kind="bar", ax=axes[0], color="steelblue", title="Distribución de RONDA"
        )
        self.df["TURNO"].value_counts().sort_index().plot(
            kind="bar", ax=axes[1], color="darkorange", title="Distribución de TURNO"
        )
        self._finish(fig, "ronda_turno_distribution.png")

    def plot_target_distribution(self):
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(self.df[TARGET_COLUMN], bins=26,
                     kde=True, color="seagreen")
        plt.title(f"Distribución de {TARGET_COLUMN}")
        self._finish(fig, "target_distribution.png")

    def plot_card_positions(self):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        for ax, col in zip(axes.ravel(), CARD_COLUMNS):
            self.df[col].value_counts().sort_index().plot(
                kind="bar", ax=ax, color="slateblue")
            ax.set_title(col)
        self._finish(fig, "card_positions.png")

    def plot_correlation_heatmap(self, feature_cols: list[str]):
        fig = plt.figure(figsize=(11, 9))
        corr = self.df[feature_cols].corr()
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Matriz de correlación entre variables predictoras")
        self._finish(fig, "correlation_heatmap.png")
        return corr

    def plot_target_correlation(self, feature_cols: list[str]):
        target_corr = (
            self.df[feature_cols + [TARGET_COLUMN]]
            .corr()[TARGET_COLUMN]
            .drop(TARGET_COLUMN)
            .sort_values(key=abs, ascending=False)
        )
        fig = plt.figure(figsize=(8, 6))
        target_corr.plot(kind="barh", color="teal")
        plt.title(f"Correlación de cada variable con {TARGET_COLUMN}")
        plt.gca().invert_yaxis()
        self._finish(fig, "target_correlation.png")
        return target_corr


class EvaluationReporter:
    """Gráficos de comparación de modelos (secciones 4 y 5 del notebook)."""

    def __init__(self, save_dir=FIGURES_DIR, save: bool = False):
        self.save_dir = save_dir
        self.save = save

    def _finish(self, fig, filename: str):
        fig.tight_layout()
        if self.save:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_dir / filename, dpi=150)
        plt.show()

    def plot_metrics_comparison(self, metrics_df: pd.DataFrame, filename="metrics_comparison.png"):
        fig, axes = plt.subplots(1, len(metrics_df.columns), figsize=(14, 4))
        for ax, metric in zip(axes, metrics_df.columns):
            metrics_df[metric].plot(kind="bar", ax=ax, title=metric)
            ax.set_xticklabels(metrics_df.index, rotation=0)
        self._finish(fig, filename)

    def plot_predicted_vs_real(self, fitted_models: dict, X_test, y_test):
        fig, axes = plt.subplots(1, len(fitted_models), figsize=(
            12, 5), sharex=True, sharey=True)
        for ax, (name, model) in zip(axes, fitted_models.items()):
            preds = model.predict(X_test)
            ax.scatter(y_test, preds, alpha=0.3, s=15)
            lims = [y_test.min(), y_test.max()]
            ax.plot(lims, lims, "r--", linewidth=1)
            ax.set_title(f"{name}: predicho vs. real")
        self._finish(fig, "predicted_vs_real.png")

    def plot_feature_importances(self, model, feature_cols: list[str], title: str):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        fig = plt.figure(figsize=(7, 6))
        importances.sort_values(ascending=True).plot(kind="barh")
        plt.title(title)
        self._finish(fig, f"feature_importances_{title.replace(' ', '_')}.png")
