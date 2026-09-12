# Guia detallada: `TrainModel` como clase padre

Esta guia elimina la duplicacion entre:

- `ml_dice_game/modeling/train_random_forest.py`
- `ml_dice_game/modeling/train_xgboost.py`

Ambos archivos hacen actualmente lo mismo: cargar datos, separar `X` e `y`, crear
CV, ejecutar `cross_validate`, ajustar el modelo, calcular metricas y guardar
archivos. Solo cambian el estimador, la clave de parametros, el nombre del modelo
y las rutas de salida.

La solucion usa el patron **Template Method**:

```text
TrainModel.run()
  cargar datos
  crear estimador              <- implementado por la hija
  validar con cross_validate
  entrenar
  evaluar en test
  guardar modelo y metricas
```

La clase padre controla el flujo. Las clases hijas solo implementan los puntos
variables del algoritmo.

## 1. Archivos que se deben modificar

| Archivo | Accion |
| --- | --- |
| `ml_dice_game/modeling/train_model.py` | Crear clase padre |
| `ml_dice_game/modeling/train_random_forest.py` | Reemplazar por clase hija |
| `ml_dice_game/modeling/train_xgboost.py` | Reemplazar por clase hija |
| `dvc.yaml` | Agregar `train_model.py` como dependencia |
| `tests/test_model_training.py` | Crear pruebas de herencia y contratos |
| `ml_dice_game/modeling/common.py` | No cambiar |
| `params.yaml` | No cambiar |
| `select_best_model.py` | No cambiar |
| `tune_best_model.py` | No cambiar |

No se debe eliminar `train_model.py` despues de la migracion. El archivo
`ml_dice_game/modeling/train.py` monolitico puede permanecer eliminado si ya se
sustituyo por los cuatro stages independientes.

## 2. Crear `train_model.py`

Crear `ml_dice_game/modeling/train_model.py` con este contenido completo:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_validate

from ml_dice_game.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_params,
)
from ml_dice_game.modeling.common import (
    SCORING,
    evaluate_model,
    load_train_test,
    make_cv,
    save_json,
    save_model,
    split_xy,
)


class TrainModel(ABC):
    """Pipeline comun para entrenar un modelo de regresion."""

    def __init__(
        self,
        train_path: Path = PROCESSED_DATA_DIR / "train.csv",
        test_path: Path = PROCESSED_DATA_DIR / "test.csv",
        model_path: Path = MODELS_DIR / "base" / "model.pkl",
        metrics_path: Path = REPORTS_DIR / "metrics" / "model.json",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.params = params if params is not None else load_params()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nombre estable usado en los JSON y reportes."""

    @property
    @abstractmethod
    def train_params_key(self) -> str:
        """Clave del estimador dentro de params['train']."""

    @abstractmethod
    def build_estimator(self) -> RegressorMixin:
        """Construye el estimador especifico de la clase hija."""

    def model_params(self) -> dict[str, Any]:
        """Combina parametros YAML y parametros comunes de sklearn."""
        return {
            **self.params["train"][self.train_params_key],
            "random_state": self.params["random_state"],
            "n_jobs": -1,
        }

    def run(self) -> dict[str, Any]:
        """Ejecuta el flujo completo y guarda los artefactos del stage."""
        train_frame, test_frame = load_train_test(
            self.train_path,
            self.test_path,
        )
        X_train, y_train = split_xy(train_frame)
        X_test, y_test = split_xy(test_frame)

        estimator = self.build_estimator()
        cv = make_cv(
            self.params["split"]["cv_n_splits"],
            self.params["random_state"],
        )
        cv_result = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
        )

        estimator.fit(X_train, y_train)
        test_metrics = evaluate_model(estimator, X_test, y_test)
        payload = self.build_metrics_payload(
            estimator,
            cv_result,
            test_metrics,
        )

        save_model(
            estimator,
            X_train.columns.tolist(),
            self.model_path,
        )
        save_json(payload, self.metrics_path)
        return payload

    def build_metrics_payload(
        self,
        estimator: RegressorMixin,
        cv_result: dict[str, Any],
        test_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Construye el formato comun consumido por select_best_model."""
        return {
            "model": self.model_name,
            **test_metrics,
            "cv_R2_mean": float(cv_result["test_r2"].mean()),
            "cv_MAE_mean": float(-cv_result["test_mae"].mean()),
            "cv_RMSE_mean": float(-cv_result["test_rmse"].mean()),
            "params": estimator.get_params(),
        }
```

### Responsabilidades del padre

`TrainModel` es dueño de todo lo que es igual para ambos algoritmos:

- leer `train.csv` y `test.csv`;
- separar la columna `PUNTAJE`;
- crear `KFold` con los parametros de `params.yaml`;
- ejecutar las tres metricas de CV;
- ajustar el modelo sobre todo `X_train`;
- evaluar contra `X_test`;
- guardar `.pkl`, `.features.json` y `.json` de metricas.

`run()` no debe duplicarse en las clases hijas. Si cambia el formato de las
metricas, se modifica una sola vez en `TrainModel`.

## 3. Reemplazar `train_random_forest.py`

Reemplazar todo el contenido de
`ml_dice_game/modeling/train_random_forest.py` por:

```python
from pathlib import Path

import typer
from sklearn.ensemble import RandomForestRegressor

from ml_dice_game.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ml_dice_game.modeling.train_model import TrainModel


class RandomForestTrainer(TrainModel):
    @property
    def model_name(self) -> str:
        return "RandomForest"

    @property
    def train_params_key(self) -> str:
        return "rf"

    def build_estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(**self.model_params())


app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "base" / "random_forest.pkl",
    metrics_path: Path = REPORTS_DIR / "metrics" / "random_forest.json",
) -> None:
    trainer = RandomForestTrainer(
        train_path=train_path,
        test_path=test_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )
    trainer.run()


if __name__ == "__main__":
    app()
```

La hija solo define:

- `model_name`: etiqueta del JSON;
- `train_params_key`: `rf`, que corresponde a `params.yaml`;
- `build_estimator`: construccion del `RandomForestRegressor`.

No debe importar `cross_validate`, `evaluate_model`, `save_json`, `save_model` ni
`SCORING`: todos pertenecen al padre.

## 4. Reemplazar `train_xgboost.py`

Reemplazar todo el contenido de
`ml_dice_game/modeling/train_xgboost.py` por:

```python
from pathlib import Path

import typer
from xgboost import XGBRegressor

from ml_dice_game.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from ml_dice_game.modeling.train_model import TrainModel


class XGBoostTrainer(TrainModel):
    @property
    def model_name(self) -> str:
        return "XGBoost"

    @property
    def train_params_key(self) -> str:
        return "xgb"

    def build_estimator(self) -> XGBRegressor:
        return XGBRegressor(**self.model_params())


app = typer.Typer()


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "base" / "xgboost.pkl",
    metrics_path: Path = REPORTS_DIR / "metrics" / "xgboost.json",
) -> None:
    trainer = XGBoostTrainer(
        train_path=train_path,
        test_path=test_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )
    trainer.run()


if __name__ == "__main__":
    app()
```

La diferencia con Random Forest es únicamente la clase del estimador y la clave
`xgb` de `params.yaml`. El resto del flujo es heredado.

## 5. Actualizar `dvc.yaml`

En el stage `train_random_forest`, agregar el archivo padre a `deps`:

```yaml
  train_random_forest:
    cmd: python -m ml_dice_game.modeling.train_random_forest
    deps:
      - data/processed/train.csv
      - data/processed/test.csv
      - ml_dice_game/modeling/common.py
      - ml_dice_game/modeling/train_model.py
      - ml_dice_game/modeling/train_random_forest.py
```

En `train_xgboost`, hacer lo mismo:

```yaml
  train_xgboost:
    cmd: python -m ml_dice_game.modeling.train_xgboost
    deps:
      - data/processed/train.csv
      - data/processed/test.csv
      - ml_dice_game/modeling/common.py
      - ml_dice_game/modeling/train_model.py
      - ml_dice_game/modeling/train_xgboost.py
```

El archivo padre es una dependencia de codigo, no una salida. Por eso no debe
aparecer en `outs` ni en `metrics`.

## 6. Crear pruebas

Crear `tests/test_model_training.py`:

```python
import pytest
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from ml_dice_game.modeling.train_model import TrainModel
from ml_dice_game.modeling.train_random_forest import RandomForestTrainer
from ml_dice_game.modeling.train_xgboost import XGBoostTrainer


@pytest.fixture
def small_params():
    return {
        "random_state": 42,
        "train": {
            "rf": {"n_estimators": 3},
            "xgb": {"n_estimators": 3},
        },
        "split": {"cv_n_splits": 2},
    }


def test_both_trainers_inherit_from_train_model():
    assert issubclass(RandomForestTrainer, TrainModel)
    assert issubclass(XGBoostTrainer, TrainModel)


def test_random_forest_child_builds_expected_estimator(small_params):
    trainer = RandomForestTrainer(params=small_params)
    estimator = trainer.build_estimator()

    assert isinstance(estimator, RandomForestRegressor)
    assert estimator.n_estimators == 3
    assert trainer.model_name == "RandomForest"


def test_xgboost_child_builds_expected_estimator(small_params):
    trainer = XGBoostTrainer(params=small_params)
    estimator = trainer.build_estimator()

    assert isinstance(estimator, XGBRegressor)
    assert estimator.n_estimators == 3
    assert trainer.model_name == "XGBoost"


def test_model_params_include_reproducibility_options(small_params):
    trainer = RandomForestTrainer(params=small_params)

    assert trainer.model_params()["random_state"] == 42
    assert trainer.model_params()["n_jobs"] == -1
```

Estas pruebas comprueban la frontera entre padre e hijas sin ejecutar un
entrenamiento costoso. Las pruebas de `run()` pueden agregarse despues usando
un `tmp_path` y datasets pequenos.

## 7. Prueba de integracion del flujo comun

Para comprobar que el padre guarda exactamente los artefactos esperados, agregar
esta prueba al mismo archivo:

```python
import json

import pandas as pd


def test_run_writes_model_and_metrics(tmp_path, small_params):
    train = pd.DataFrame({
        "RONDA": [1, 1, 2, 2, 3, 3],
        "feature": [1, 2, 3, 4, 5, 6],
        "PUNTAJE": [2, 3, 4, 5, 6, 7],
    })
    test = pd.DataFrame({
        "RONDA": [1, 2],
        "feature": [7, 8],
        "PUNTAJE": [8, 9],
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "models" / "rf.pkl"
    metrics_path = tmp_path / "metrics" / "rf.json"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    trainer = RandomForestTrainer(
        train_path=train_path,
        test_path=test_path,
        model_path=model_path,
        metrics_path=metrics_path,
        params=small_params,
    )
    payload = trainer.run()

    assert model_path.exists()
    assert model_path.with_suffix(".features.json").exists()
    assert metrics_path.exists()
    assert payload["model"] == "RandomForest"
    assert set(("R2", "MAE", "RMSE")).issubset(payload)
    assert json.loads(metrics_path.read_text())["model"] == "RandomForest"
```

## 8. Eliminar duplicacion

Despues de reemplazar los dos archivos, verificar lo siguiente:

- `cross_validate` aparece solo en `train_model.py` y no en los hijos;
- `evaluate_model` aparece solo en `train_model.py` para estos stages;
- `save_model` y `save_json` aparecen solo en `train_model.py` para estos stages;
- los hijos no leen `params.yaml` directamente;
- ambos comandos conservan las mismas rutas de salida;
- `select_best_model.py` sigue leyendo los mismos JSON.

No mover `select_best_model.py` ni `tune_best_model.py` a esta jerarquia. Esos
modulos trabajan con modelos ya entrenados y no son variantes del entrenamiento
base.

## 9. Validacion final

Ejecutar en este orden desde la raiz del repositorio:

```text
python -m pytest tests/test_model_training.py tests/test_modeling.py
python -m ml_dice_game.modeling.train_random_forest
python -m ml_dice_game.modeling.train_xgboost
python -m ml_dice_game.modeling.select_best_model
python -m ml_dice_game.modeling.tune_best_model
python -m pytest tests
python -m dvc repro
```

La refactorizacion es correcta cuando:

1. ambos trainers pasan las pruebas de herencia;
2. ambos stages generan sus propios modelos y metricas;
3. los dos JSON tienen las mismas claves de contrato;
4. `selection.json` puede escoger cualquiera de los dos modelos;
5. el stage de tuning sigue publicando `models/model.pkl`;
6. DVC vuelve a ejecutar RF y XGBoost si cambia `train_model.py`;
7. DVC no mezcla la salida de los dos stages.
