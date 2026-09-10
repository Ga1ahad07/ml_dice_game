import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger


class ShapExplainer:
    """Interpretabilidad del modelo final con SHAP (sección 6 del notebook)."""

    def __init__(self, model, X_test: pd.DataFrame):
        self.model = model
        self.X_test = X_test
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer(X_test)
        logger.info(f"SHAP values shape: {self.shap_values.values.shape}")

    def plot_global_importance(self, plot_type: str = "bar"):
        shap.summary_plot(self.shap_values, self.X_test,
                          plot_type=plot_type, show=False)
        plt.tight_layout()
        plt.show()

    def plot_beeswarm(self):
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.tight_layout()
        plt.show()

    def top_features(self, n: int = 3) -> list[str]:
        return (
            pd.Series(np.abs(self.shap_values.values).mean(
                axis=0), index=self.X_test.columns)
            .sort_values(ascending=False)
            .head(n)
            .index.tolist()
        )

    def plot_dependence(self, features: list[str]):
        fig, axes = plt.subplots(1, len(features), figsize=(16, 4))
        axes = [axes] if len(features) == 1 else axes
        for ax, feat in zip(axes, features):
            shap.dependence_plot(feat, self.shap_values.values,
                                 self.X_test, ax=ax, show=False)
            ax.set_title(f"Dependence plot: {feat}")
        plt.tight_layout()
        plt.show()

    def plot_local_explanation(self, index: int):
        shap.plots.waterfall(self.shap_values[index], show=False)
        plt.tight_layout()
        plt.show()
