# tests/test_dataset.py
import pandas as pd
from ml_dice_game.dataset import DiceGameDataset


def test_validate_game_rules_detects_invalid_sum(tmp_path):
    df = pd.DataFrame({
        "C1A": [1], "C1B": [1], "C1C": [1], "C2A": [1], "C2B": [1],
        "C2C": [1], "C3A": [1], "C3B": [1], "C3C": [1],  # suma = 9, inválido
    })
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, sep=";", index=False)

    ds = DiceGameDataset(path=csv_path)
    result = ds.validate_game_rules()
    assert result["invalid_sum_rows"] == 1
