# tests/test_features.py
import pandas as pd
from ml_dice_game.features import DataSplitter


def test_split_preserves_stratification():
    df = pd.DataFrame({
        "RONDA": [1, 1, 1, 1, 2, 2, 2, 2] * 5,
        "PUNTAJE": range(40),
    })
    splitter = DataSplitter(stratify_col="RONDA")
    X_train, X_test, y_train, y_test, _ = splitter.split(
        df, target_col="PUNTAJE")
    assert len(X_train) + len(X_test) == len(df)
