from dataclasses import dataclass

from sklearn.model_selection import RandomizedSearchCV

from ml_dice_game.config import RANDOM_STATE, load_params


@dataclass
class ModelTuner:
    estimator: object
    param_distributions: dict
    cv: object
    n_iter: int = 50
    scoring: str = "roc_auc"

    @classmethod
    def from_config(cls, estimator, cv):
        params = load_params().get("tune", {})
        model_key = "xgb_param_distributions" if estimator.__class__.__name__ == "XGBClassifier" else "rf_param_distributions"
        return cls(
            estimator=estimator,
            param_distributions=params.get(model_key, {}),
            cv=cv,
            n_iter=int(params.get("n_iter", 50)),
            scoring=params.get("scoring", "roc_auc"),
        )

    def fit(self, X, y):
        search = RandomizedSearchCV(
            self.estimator,
            self.param_distributions,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=self.cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        return search.fit(X, y)
