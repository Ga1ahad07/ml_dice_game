from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


class ShapExplainer:
    """Explica un clasificador de arbol para la clase positiva de VENTAJA."""

    def __init__(self, model, X_test: pd.DataFrame, positive_class: int = 1):
        self.X_test = X_test
        self.positive_class = positive_class
        raw_explanation = shap.TreeExplainer(model)(X_test)
        self.explanation = self._select_positive_class(raw_explanation)

    def _select_positive_class(self, explanation: shap.Explanation) -> shap.Explanation:
        """Normaliza la salida de SHAP a los valores de VENTAJA=1."""
        if explanation.values.ndim == 3:
            return explanation[..., self.positive_class]
        return explanation

    def save_global_importance(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(
            self.explanation,
            self.X_test,
            plot_type="bar",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_global_beeswarm(self, path: Path) -> None:
        """Guarda el impacto global de las variables sobre VENTAJA=1."""
        path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(
            self.explanation,
            self.X_test,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_dependence_plot(self, feature: str, path: Path) -> None:
        """Guarda la dependencia SHAP de una variable sobre VENTAJA=1."""
        if feature not in self.X_test.columns:
            raise ValueError(f"La feature '{feature}' no existe en X_test")

        path.parent.mkdir(parents=True, exist_ok=True)
        shap.dependence_plot(
            feature,
            self.explanation.values,
            self.X_test,
            show=False,
        )
        plt.title(f"Dependencia SHAP: {feature} sobre VENTAJA=1")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_local_waterfall(self, index: int, path: Path) -> None:
        """Guarda la explicación local de una fila para VENTAJA=1."""
        if index < 0 or index >= len(self.X_test):
            raise IndexError(f"El indice {index} no existe en X_test")

        path.parent.mkdir(parents=True, exist_ok=True)
        shap.plots.waterfall(self.explanation[index], show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
