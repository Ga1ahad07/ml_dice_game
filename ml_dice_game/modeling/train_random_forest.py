from sklearn.ensemble import RandomForestClassifier

from ml_dice_game.config import RANDOM_STATE, load_params


class RandomForestFactory:
    name = "RandomForest"

    @staticmethod
    def create(**overrides):
        params = load_params().get("train", {}).get("rf", {}).copy()
        params.update(overrides)
        return RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, **params
        )
