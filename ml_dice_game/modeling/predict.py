import json
from pathlib import Path

import joblib
import pandas as pd


class ModelPredictor:
    def __init__(self, model_path: Path, features_path: Path):
        self.model = joblib.load(model_path)
        self.features = json.loads(features_path.read_text(encoding="utf-8"))

    def predict(self, values: dict) -> dict:
        missing = [feature for feature in self.features if feature not in values]
        if missing:
            raise ValueError(f"Faltan features: {missing}")
        frame = pd.DataFrame(
            [[values[feature] for feature in self.features]], columns=self.features)
        prediction = int(self.model.predict(frame)[0])
        probability = float(self.model.predict_proba(frame)[0, 1])
        return {"prediction": prediction, "probability": probability}
