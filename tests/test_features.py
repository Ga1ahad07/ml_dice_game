# tests/test_features.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ml_dice_game.config import TARGET_COLUMN
from ml_dice_game.features import DataSplitter


def test_split_preserves_stratification():
    df = pd.DataFrame({
        "RONDA": [1, 1, 1, 1, 2, 2, 2, 2] * 5,
        "VENTAJA": [0, 1] * 20,
    })
    splitter = DataSplitter(diagnostic_col="RONDA")
    X_train, X_test, y_train, y_test, _ = splitter.split(
        df, target_col=TARGET_COLUMN)
    assert len(X_train) + len(X_test) == len(df)
    assert "VENTAJA" not in X_train.columns
    assert y_train.value_counts(normalize=True).sort_index().to_dict() == {
        0: 0.5,
        1: 0.5,
    }
    assert y_test.value_counts(normalize=True).sort_index().to_dict() == {
        0: 0.5,
        1: 0.5,
    }


def test_get_cv_returns_stratified_kfold():
    cv = DataSplitter(diagnostic_col="RONDA").get_cv()

    assert isinstance(cv, StratifiedKFold)
