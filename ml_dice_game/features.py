from pathlib import Path
import typer
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml_dice_game.config import (
    CV_N_SPLITS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)


class DataSplitter:

    def __init__(
        self,
        diagnostic_col: str = "RONDA",
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        cv_splits: int = CV_N_SPLITS,
    ):
        self.diagnostic_col = diagnostic_col
        self.test_size = test_size
        self.random_state = random_state
        self.cv_splits = cv_splits

    def split(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN):
        feature_cols = [c for c in df.columns if c != target_col]
        X, y = df[feature_cols].copy(), df[target_col].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_cols

    def check_stratification(self, df, X_train, X_test):
        return {
            "target_full": df[TARGET_COLUMN].value_counts(normalize=True).sort_index(),
            "target_train": df.loc[X_train.index, TARGET_COLUMN]
            .value_counts(normalize=True).sort_index(),
            "target_test": df.loc[X_test.index, TARGET_COLUMN]
            .value_counts(normalize=True).sort_index(),
            "diagnostic_full": df[self.diagnostic_col].value_counts(normalize=True).sort_index(),
            "diagnostic_train": df.loc[X_train.index, self.diagnostic_col]
            .value_counts(normalize=True).sort_index(),
            "diagnostic_test": df.loc[X_test.index, self.diagnostic_col]
            .value_counts(normalize=True).sort_index(),
        }

    def get_cv(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )


app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset2.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test.csv",
    diagnostic_col: str = "RONDA",
):
    logger.info(f"Cargando dataset desde {input_path}")
    df = pd.read_csv(input_path, sep=";")

    splitter = DataSplitter(diagnostic_col=diagnostic_col)
    X_train, X_test, y_train, y_test, _ = splitter.split(
        df,
        target_col=TARGET_COLUMN,
    )

    # Este diagnostico comprueba RONDA; el split ya fue estratificado por VENTAJA.
    strat_check = splitter.check_stratification(df, X_train, X_test)
    logger.info(
        f"Proporcion de VENTAJA en train:\n{strat_check['target_train']}"
    )
    logger.info(
        f"Proporcion de VENTAJA en test:\n{strat_check['target_test']}"
    )
    logger.info(
        f"Proporcion diagnostica de {diagnostic_col} en train:\n"
        f"{strat_check['diagnostic_train']}"
    )
    logger.info(
        f"Proporcion diagnostica de {diagnostic_col} en test:\n"
        f"{strat_check['diagnostic_test']}"
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    logger.success(f"Train guardado en {train_output_path} ({train_df.shape})")
    logger.success(f"Test guardado en {test_output_path} ({test_df.shape})")


if __name__ == "__main__":
    app()
