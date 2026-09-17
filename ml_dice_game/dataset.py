from pathlib import Path
import pandas as pd
from loguru import logger
import typer
import json

from ml_dice_game.config import (
    CARD_COLUMNS, BOARD_COLUMNS, TARGET_COLUMN, CARD_SUM_EXPECTED, CARD_NONZERO_EXPECTED, RAW_DATA_DIR, INTERIM_DATA_DIR
)


class DiceGameDataset:
    """Carga y valida el dataset crudo del juego de mesa."""

    def __init__(self, path: Path = RAW_DATA_DIR / "dataset2.csv", sep: str = ";"):
        self.path = path
        self.sep = sep
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df

    def load(self) -> pd.DataFrame:
        logger.info(f"Cargando dataset desde {self.path}")
        self._df = pd.read_csv(self.path, sep=self.sep)
        logger.success(f"Dataset cargado. Shape: {self._df.shape}")
        return self._df

    def summary(self) -> dict:
        return {
            "dtypes": self.df.dtypes,
            "n_missing": self.df.isna().sum().sum(),
            "n_duplicates": self.df.duplicated().sum(),
            "describe": self.df.describe().T,
        }

    def validate_game_rules(self) -> dict:
        """Reglas de negocio del juego."""
        card_sum = self.df[CARD_COLUMNS].sum(axis=1)
        n_nonzero = (self.df[CARD_COLUMNS] != 0).sum(axis=1)

        invalid_sum = int((card_sum != CARD_SUM_EXPECTED).sum())
        invalid_nonzero = int((n_nonzero != CARD_NONZERO_EXPECTED).sum())

        required_columns = CARD_COLUMNS + BOARD_COLUMNS + \
            ["RONDA", "TURNO", TARGET_COLUMN]
        missing_columns = sorted(set(required_columns) - set(self.df.columns))
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

        invalid_target = int(
            self.df[TARGET_COLUMN].isna().sum()
            + (~self.df[TARGET_COLUMN].isin([0, 1])).sum()
        )

        board_values = self.df[BOARD_COLUMNS]
        invalid_board = int(
            ((board_values < -6) | (board_values > 6)).any(axis=1).sum()
        )

        if invalid_sum or invalid_nonzero:
            logger.warning(
                f"Filas con suma != {CARD_SUM_EXPECTED}: {invalid_sum} | "
                f"Filas con no-ceros != {CARD_NONZERO_EXPECTED}: {invalid_nonzero}"
            )
        else:
            logger.success("Todas las filas cumplen las reglas del juego.")

        return {
            "invalid_sum_rows": invalid_sum,
            "invalid_nonzero_rows": invalid_nonzero,
            "invalid_target_rows": invalid_target,
            "invalid_board_rows": invalid_board,
        }


app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset2.csv",
    output_path: Path = INTERIM_DATA_DIR / "validation_report.json",
):
    logger.info("Validando dataset...")

    dataset = DiceGameDataset(path=input_path)
    dataset.load()

    summary = dataset.summary()
    rules = dataset.validate_game_rules()

    report = {
        "shape": list(dataset.df.shape),
        "n_missing": int(summary["n_missing"]),
        "n_duplicates": int(summary["n_duplicates"]),
        "invalid_sum_rows": rules["invalid_sum_rows"],
        "invalid_nonzero_rows": rules["invalid_nonzero_rows"],
        "invalid_target_rows": rules["invalid_target_rows"],
        "invalid_board_rows": rules["invalid_board_rows"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.success(f"Reporte de validación guardado en {output_path}")


if __name__ == "__main__":
    app()
