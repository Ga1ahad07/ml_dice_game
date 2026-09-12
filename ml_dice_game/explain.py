from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


class ShapExplainer:
    def __init__(self, model, X_test: pd.DataFrame):
        self.X_test = X_test
        self.explanation = shap.TreeExplainer(model)(X_test)

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
