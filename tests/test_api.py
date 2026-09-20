import json

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from api import main

FEATURES = [
    "C1A", "C1B", "C1C", "C2A", "C2B", "C2C", "C3A", "C3B", "C3C",
    "T1A", "T1B", "T1C", "T2A", "T2B", "T2C", "T3A", "T3B", "T3C",
    "RONDA", "TURNO",
]


def test_predict_endpoint_returns_probability(tmp_path, monkeypatch):
    model = DummyClassifier(strategy="prior")
    model.fit(np.zeros((4, 20)), [0, 1, 0, 1])
    model_path = tmp_path / "model.joblib"
    features_path = tmp_path / "features.json"
    joblib.dump(model, model_path)
    features_path.write_text(json.dumps(FEATURES), encoding="utf-8")

    monkeypatch.setattr(main, "service", main.ModelService(
        model_path, features_path))
    client = TestClient(main.app)
    response = client.post(
        "/predict", json={feature: 0 for feature in FEATURES})

    assert response.status_code == 200
    assert set(response.json()) == {"prediction", "probability"}
    assert 0 <= response.json()["probability"] <= 1


def test_predict_endpoint_rejects_missing_feature():
    client = TestClient(main.app)
    response = client.post("/predict", json={})
    assert response.status_code == 422
