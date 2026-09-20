from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


class ShapExplainer:
    """Explica un clasificador usando únicamente los datos proporcionados."""

    def __init__(
        self,
        model,
        X_data: pd.DataFrame,
        positive_class: int = 1,
    ):
        self.X_data = X_data
        self.positive_class = positive_class

        raw_explanation = shap.TreeExplainer(model)(X_data)
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
            self.X_data,
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
            self.X_data,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_group_importance(self, group_map: dict[str, str], path: Path) -> None:
        """Guarda la importancia SHAP agregada por grupo de variables."""
        missing = set(self.X_data.columns) - set(group_map)
        if missing:
            raise ValueError(f"No hay grupo SHAP para: {sorted(missing)}")

        importance = pd.Series(
            abs(self.explanation.values).sum(axis=0),
            index=self.X_data.columns,
        )
        grouped = importance.groupby(importance.index.map(group_map)).sum()
        path.parent.mkdir(parents=True, exist_ok=True)
        ax = grouped.sort_values().plot(kind="barh", figsize=(7, 4), color="teal")
        ax.set_title("Importancia SHAP por grupo de variable")
        ax.set_xlabel("Sumatoria de |valor SHAP|")
        ax.set_ylabel("Grupo")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_dependence_plot(self, feature: str, path: Path) -> None:
        """Guarda la dependencia SHAP de una variable sobre VENTAJA=1."""
        if feature not in self.X_data.columns:
            raise ValueError(f"La feature '{feature}' no existe en X_data")

        path.parent.mkdir(parents=True, exist_ok=True)
        shap.dependence_plot(
            feature,
            self.explanation.values,
            self.X_data,
            show=False,
        )
        plt.title(f"Dependencia SHAP: {feature} sobre VENTAJA=1")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def save_local_waterfall(self, index: int, path: Path) -> None:
        """Guarda la explicación local de una fila para VENTAJA=1."""
        if index < 0 or index >= len(self.X_data):
            raise IndexError(f"El indice {index} no existe en X_data")

        path.parent.mkdir(parents=True, exist_ok=True)
        shap.plots.waterfall(self.explanation[index], show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
