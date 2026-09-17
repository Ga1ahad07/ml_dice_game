# tests/test_dataset.py
import pandas as pd
from ml_dice_game.dataset import DiceGameDataset
from ml_dice_game.config import BOARD_COLUMNS, CARD_COLUMNS


def make_valid_frame(target_values=(0, 1)):
    frame = {column: [0] * len(target_values) for column in CARD_COLUMNS}
    frame[CARD_COLUMNS[0]] = [4] * len(target_values)
    frame[CARD_COLUMNS[1]] = [4] * len(target_values)
    frame.update({column: [0] * len(target_values)
                 for column in BOARD_COLUMNS})
    frame["RONDA"] = list(range(1, len(target_values) + 1))
    frame["TURNO"] = [1] * len(target_values)
    frame["VENTAJA"] = list(target_values)
    return pd.DataFrame(frame)


def test_validate_game_rules_detects_invalid_sum(tmp_path):
    df = make_valid_frame((0,))
    df.loc[0, "C1A"] = 1  # suma = 5, invalido
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, sep=";", index=False)

    ds = DiceGameDataset(path=csv_path)
    result = ds.validate_game_rules()
    assert result["invalid_sum_rows"] == 1


def test_validate_game_rules_detects_invalid_target(tmp_path):
    df = make_valid_frame((2,))
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, sep=";", index=False)

    result = DiceGameDataset(path=csv_path).validate_game_rules()

    assert result["invalid_target_rows"] == 1


def test_validate_game_rules_detects_invalid_board(tmp_path):
    df = make_valid_frame((1,))
    df.loc[0, BOARD_COLUMNS[0]] = 7
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, sep=";", index=False)

    result = DiceGameDataset(path=csv_path).validate_game_rules()

    assert result["invalid_board_rows"] == 1


def test_validate_game_rules_rejects_missing_columns(tmp_path):
    df = make_valid_frame((0,)).drop(columns=["T1A"])
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, sep=";", index=False)

    try:
        DiceGameDataset(path=csv_path).validate_game_rules()
    except ValueError as error:
        assert "T1A" in str(error)
    else:
        raise AssertionError("Se esperaba un error por columna ausente")
