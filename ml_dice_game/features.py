from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer


from sklearn.model_selection import train_test_split, KFold
from loguru import logger

from ml_dice_game.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE, CV_N_SPLITS, TARGET_COLUMN


class DataSplitter:
    """Split train/test estratificado por RONDA + KFold para CV (sección 2 del notebook)."""

    def __init__(
        self,
        stratify_col: str,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        cv_splits: int = CV_N_SPLITS,
    ):
        self.stratify_col = stratify_col
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
            stratify=df[self.stratify_col],
        )
        logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_cols

    def check_stratification(self, df, X_train, X_test):
        """Reemplaza los prints de la celda 19 que verifican proporciones por RONDA."""
        return {
            "full": df[self.stratify_col].value_counts(normalize=True).sort_index(),
            "train": df.loc[X_train.index, self.stratify_col].value_counts(normalize=True).sort_index(),
            "test": df.loc[X_test.index, self.stratify_col].value_counts(normalize=True).sort_index(),
        }

    def get_cv(self) -> KFold:
        return KFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)


app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test.csv",
    stratify_col: str = "RONDA",
):
    logger.info(f"Cargando dataset desde {input_path}")
    df = pd.read_csv(input_path, sep=";")

    splitter = DataSplitter(stratify_col=stratify_col)
    X_train, X_test, y_train, y_test, feature_cols = splitter.split(df)

    # Verificación de estratificación (opcional, solo para log)
    strat_check = splitter.check_stratification(df, X_train, X_test)
    logger.info(
        f"Proporciones por '{stratify_col}' — train:\n{strat_check['train']}")
    logger.info(
        f"Proporciones por '{stratify_col}' — test:\n{strat_check['test']}")

    # Reconstruir dataframes completos (features + target) para guardar
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
