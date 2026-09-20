from xgboost import XGBClassifier

from ml_dice_game.config import RANDOM_STATE, load_params


class XGBoostFactory:
    name = "XGBoost"

    @staticmethod
    def create(**overrides):
        params = load_params().get("train", {}).get("xgb", {}).copy()
        params.update(overrides)
        return XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
