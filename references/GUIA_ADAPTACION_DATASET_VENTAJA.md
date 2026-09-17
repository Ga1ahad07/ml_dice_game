# Guia de adaptacion del proyecto al notebook `Notebook_VENTAJA`

## 1. Alcance

Esta guia indica como adaptar la refactorizacion POO al dataset y objetivo usados por `notebooks/Notebook_VENTAJA.ipynb`.

El notebook define un problema de **clasificacion binaria**:

- objetivo: `VENTAJA`;
- clases: `0` y `1`;
- modelos: `RandomForestClassifier` y `XGBClassifier`;
- validacion: `StratifiedKFold`;
- metricas: `Accuracy`, `Precision`, `Recall`, `F1` y `ROC_AUC`;
- salida: clase predicha y probabilidad de `VENTAJA=1`.

No modificar el notebook. Esta guia debe permanecer en `references/` y no debe agregarse a `docs/mkdocs.yml`.

## 2. Estado objetivo del pipeline

El pipeline POO debe quedar con este flujo:

```text
data/raw/dataset2.csv
        |
        v
validate_data -> data/interim/validation_report.json
        |
        v
split_data -> data/processed/train.csv + test.csv
        |
        +--> RandomForestClassifier -> random_forest.pkl + metricas
        |
        +--> XGBClassifier         -> xgboost.pkl + metricas
        |
        v
select_best_model -> mayor ROC_AUC
        |
        v
tune_best_model -> modelo final clasificador
        |
        v
API -> clase predicha + probabilidad de VENTAJA=1
```

Los cambios deben hacerse por archivo, como se detalla a continuacion.

## 3. `ml_dice_game/config.py`

### Estado actual

Este archivo ya contiene:

```python
TARGET_COLUMN = "VENTAJA"
BOARD_COLUMNS = [
    "T1A", "T1B", "T1C", "T2A", "T2B", "T2C",
    "T3A", "T3B", "T3C",
]
```

Tambien conserva constantes de split y rutas. No volver a cambiar `TARGET_COLUMN` a `PUNTAJE`.

### Cambios exactos

1. Mantener `TARGET_COLUMN = "VENTAJA"`.
2. Mantener `CARD_COLUMNS` y `BOARD_COLUMNS`.
3. Mantener `RANDOM_STATE`, `TEST_SIZE` y `CV_N_SPLITS`.
4. Mantener los diccionarios de hiperparametros, pero usarlos con clasificadores.
5. Agregar, si se quiere centralizar la configuracion, las metricas de clasificación:

```python
CLASSIFICATION_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
```

### No modificar aqui

No importar `ClassifierMixin` en `config.py`. Los tipos de los estimadores pertenecen a los modulos de modelado, no al archivo de constantes.

## 4. `ml_dice_game/dataset.py`

### Estado actual

El archivo ya usa por defecto:

```python
RAW_DATA_DIR / "dataset2.csv"
```

y separador `;`, pero `validate_game_rules()` solo valida las cartas.

### Cambios exactos

Modificar `DiceGameDataset.validate_game_rules()` para agregar, en este orden:

1. Validacion de columnas requeridas.
2. Validacion de `VENTAJA`.
3. Validacion de las cartas existente.
4. Validacion del rango del tablero.

Agregar los imports correspondientes desde `config.py`:

```python
from ml_dice_game.config import (
    BOARD_COLUMNS,
    CARD_COLUMNS,
    CARD_NONZERO_EXPECTED,
    CARD_SUM_EXPECTED,
    TARGET_COLUMN,
)
```

Agregar dentro de `validate_game_rules()`:

```python
required_columns = CARD_COLUMNS + BOARD_COLUMNS + ["RONDA", "TURNO", TARGET_COLUMN]
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
```

Ampliar el diccionario de retorno:

```python
return {
    "invalid_sum_rows": invalid_sum,
    "invalid_nonzero_rows": invalid_nonzero,
    "invalid_target_rows": invalid_target,
    "invalid_board_rows": invalid_board,
}
```

En `main()`, agregar `invalid_target_rows` e `invalid_board_rows` al JSON de `validation_report.json`.

### Eliminar en este archivo

Eliminar imports no utilizados como `tqdm`, `load_params` y cualquier import duplicado. No agregar aqui validacion de metricas del modelo.

## 5. `ml_dice_game/features.py`

### Estado actual

El split actual usa `stratify_col`, cuyo valor por defecto es `RONDA`, y el CV actual devuelve `KFold`.

### Cambios exactos

1. Mantener `target_col=TARGET_COLUMN` para que `VENTAJA` no entre en las features.
2. Cambiar el split para estratificar por `VENTAJA`, siguiendo el notebook:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=self.test_size,
    random_state=self.random_state,
    stratify=y,
)
```

3. Cambiar el import:

```python
from sklearn.model_selection import StratifiedKFold, train_test_split
```

4. Cambiar `get_cv()`:

```python
def get_cv(self) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=self.cv_splits,
        shuffle=True,
        random_state=self.random_state,
    )
```

5. Mantener `check_stratification()` unicamente como diagnostico. En `check_stratification()`, comparar la distribucion de `RONDA` entre el dataset completo, train y test, y agregar tambien la distribucion de `VENTAJA` para comprobar que el split quedo balanceado. Esta funcion no debe decidir como se divide el dataset.
6. En el constructor `DataSplitter.__init__()`, renombrar el parametro `stratify_col` a `diagnostic_col` para evitar la ambiguedad, o conservar el nombre antiguo solo por compatibilidad. Si se conserva, su valor debe ser `"RONDA"` y debe usarse solamente dentro de `check_stratification()`.

El cambio concreto recomendado en `DataSplitter` es:

```python
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
```

En `DataSplitter.split()`, la unica columna usada en `stratify` debe ser `y`, que contiene `VENTAJA`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=self.test_size,
    random_state=self.random_state,
    stratify=y,
)
```

En `DataSplitter.check_stratification()`, usar `self.diagnostic_col` solo para generar el diagnostico de `RONDA`:

```python
def check_stratification(self, df, X_train, X_test):
    return {
        "target_full": df[TARGET_COLUMN].value_counts(normalize=True).sort_index(),
        "target_train": df.loc[X_train.index, TARGET_COLUMN]
        .value_counts(normalize=True).sort_index(),
        "target_test": df.loc[X_test.index, TARGET_COLUMN]
        .value_counts(normalize=True).sort_index(),
        "diagnostic_full": df[self.diagnostic_col]
        .value_counts(normalize=True).sort_index(),
        "diagnostic_train": df.loc[X_train.index, self.diagnostic_col]
        .value_counts(normalize=True).sort_index(),
        "diagnostic_test": df.loc[X_test.index, self.diagnostic_col]
        .value_counts(normalize=True).sort_index(),
    }
```

En `main()` de `features.py`, cambiar la instancia y los logs para que quede claro que `RONDA` es diagnostica. El `main()` completo recomendado debe quedar asi:

```python
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
```

En este `main()`, `diagnostic_col` solo se pasa a `check_stratification()`. La estratificacion real ocurre dentro de `DataSplitter.split()` mediante `stratify=y`, donde `y` es `VENTAJA`.

Si se desea conservar el nombre publico actual del argumento CLI, usar `stratify_col` en lugar de `diagnostic_col` en la firma y en la instancia, pero mantener exactamente la misma responsabilidad diagnostica:

```python
def main(
    input_path: Path = RAW_DATA_DIR / "dataset2.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test.csv",
    stratify_col: str = "RONDA",
):
    splitter = DataSplitter(diagnostic_col=stratify_col)
```

El nombre `stratify_col` sería entonces heredado por compatibilidad, pero no debe aparecer en el argumento `stratify=` de `train_test_split()`.

La version resumida anterior queda reemplazada por el bloque completo:

```python
splitter = DataSplitter(diagnostic_col="RONDA")
strat_check = splitter.check_stratification(df, X_train, X_test)
logger.info(f"Proporcion de VENTAJA en train:\n{strat_check['target_train']}")
logger.info(f"Proporcion de VENTAJA en test:\n{strat_check['target_test']}")
logger.info(f"Proporcion diagnostica de RONDA en train:\n{strat_check['diagnostic_train']}")
logger.info(f"Proporcion diagnostica de RONDA en test:\n{strat_check['diagnostic_test']}")
```

Si no se quiere renombrar el parametro publico, conservar `stratify_col="RONDA"` en `__init__()` y en `main()`, pero cambiar su nombre semantico en la documentacion: es una columna diagnostica, no la columna usada en `train_test_split`. No cambiarlo a `VENTAJA` para luego seguir usando `df[self.stratify_col]`; eso duplicaria la responsabilidad de `y` y haria menos claro el codigo.

### Eliminar en este archivo

Eliminar el uso de `KFold` como validacion cruzada principal. No crear una columna auxiliar de `VENTAJA` dentro de las features.

## 6. `ml_dice_game/modeling/common.py`

### Estado actual

Este archivo es de regresion: importa `RegressorMixin`, `KFold`, `R2`, `MAE` y `RMSE`.

### Cambios exactos

1. Reemplazar imports:

```python
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
```

2. Reemplazar `SCORING`:

```python
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
```

3. Cambiar `make_cv()`:

```python
def make_cv(cv_n_splits: int, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=cv_n_splits,
        shuffle=True,
        random_state=random_state,
    )
```

4. Reemplazar `evaluate_model()`:

```python
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1": float(f1_score(y_test, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_test, probabilities)),
    }
```

5. Mantener sin cambios `load_train_test`, `split_xy`, `save_json`, `save_model` y `load_model`, porque no dependen del tipo de estimador.

### Eliminar en este archivo

Eliminar imports y uso de:

- `RegressorMixin`;
- `KFold`;
- `mean_absolute_error`;
- `mean_squared_error`;
- `r2_score`;
- `R2`, `MAE`, `RMSE` y sus scorers.

## 7. `ml_dice_game/modeling/train_model.py`

### Estado actual

La clase `TrainModel` declara `RegressorMixin` y crea el CV con `make_cv()`.

### Cambios exactos

1. Cambiar:

```python
from sklearn.base import RegressorMixin
```

por:

```python
from sklearn.base import ClassifierMixin
```

2. Cambiar la firma abstracta:

```python
def build_estimator(self) -> ClassifierMixin:
```

3. En `run()`, mantener `cross_validate`, pero obtener `SCORING` y `make_cv` desde `ml_dice_game.modeling.common`. `common.py` es el unico lugar que debe construir el `StratifiedKFold`; `train_model.py` solo lo solicita y lo pasa a scikit-learn.

En el bloque de imports de `train_model.py`, confirmar que existan:

```python
from sklearn.model_selection import cross_validate

from ml_dice_game.modeling.common import (
    SCORING,
    evaluate_model,
    load_train_test,
    make_cv,
    save_json,
    save_model,
    split_xy,
)
```

No importar `KFold` ni crear directamente un `StratifiedKFold` dentro de `train_model.py`. La implementacion de `make_cv()` debe estar en `common.py` y devolverlo:

```python
def make_cv(cv_n_splits: int, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=cv_n_splits,
        shuffle=True,
        random_state=random_state,
    )
```

Dentro de `TrainModel.run()`, reemplazar el bloque actual de validacion cruzada por este:

```python
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
```

La responsabilidad de cada elemento es:

- `make_cv(...)`: crea los cinco folds estratificados por `VENTAJA`.
- `cv=cv`: entrega esos folds a `cross_validate`.
- `SCORING`: solicita simultaneamente `accuracy`, `precision`, `recall`, `f1` y `roc_auc`.
- `cross_validate(...)`: entrena y evalua el clasificador en cada fold.

El resultado `cv_result` debe contener las claves `test_accuracy`, `test_precision`, `test_recall`, `test_f1` y `test_roc_auc`, que luego consume `build_metrics_payload()`.
4. Cambiar la firma de `build_metrics_payload()` a `ClassifierMixin`.
5. Reemplazar el payload de regresion por:

```python
return {
    "target": TARGET_COLUMN,
    "model": self.model_name,
    **test_metrics,
    "cv_Accuracy_mean": float(cv_result["test_accuracy"].mean()),
    "cv_Precision_mean": float(cv_result["test_precision"].mean()),
    "cv_Recall_mean": float(cv_result["test_recall"].mean()),
    "cv_F1_mean": float(cv_result["test_f1"].mean()),
    "cv_ROC_AUC_mean": float(cv_result["test_roc_auc"].mean()),
    "params": estimator.get_params(),
}
```

### Eliminar en este archivo

Eliminar `cv_R2_mean`, `cv_MAE_mean` y `cv_RMSE_mean`. No introducir umbrales ni logica de API en esta clase: su responsabilidad es entrenar y evaluar.

## 8. `ml_dice_game/modeling/train_random_forest.py`

### Cambios exactos

1. Cambiar el import:

```python
from sklearn.ensemble import RandomForestClassifier
```

2. Cambiar el tipo de retorno y la construccion:

```python
def build_estimator(self) -> RandomForestClassifier:
    return RandomForestClassifier(**self.model_params())
```

3. Mantener `model_name`, `train_params_key`, rutas y herencia de `TrainModel`.

### No modificar aqui

No implementar entrenamiento manual, umbral ni calculo de métricas en esta clase. Esos comportamientos pertenecen a la clase base y a `common.py`.

## 9. `ml_dice_game/modeling/train_xgboost.py`

### Cambios exactos

1. Cambiar el import:

```python
from xgboost import XGBClassifier
```

2. Cambiar la construccion:

```python
def build_estimator(self) -> XGBClassifier:
    return XGBClassifier(**self.model_params())
```

3. Mantener `model_params()` sin agregar parametros en este archivo. `train_xgboost.py` solo construye `XGBClassifier`; la configuracion `objective` y `eval_metric` se define en `params.yaml` y llega al estimador mediante `self.model_params()`.

### No modificar aqui

No poner aqui la selección del mejor modelo ni el tuning; este archivo solo construye el estimador XGBoost base.

## 10. `ml_dice_game/modeling/select_best_model.py`

### Cambios exactos

1. Cambiar el conjunto de campos leidos desde cada JSON a:

```python
("Accuracy", "Precision", "Recall", "F1", "ROC_AUC")
```

2. Cambiar:

```python
selected_model = metrics_df["R2"].idxmax()
```

por:

```python
selected_model = metrics_df["ROC_AUC"].idxmax()
```

3. Cambiar el JSON de selección:

```python
{
    "target": "VENTAJA",
    "selected_model": selected_model,
    "model_path": str(...),
    "selection_metric": "ROC_AUC",
    "metrics": metrics,
}
```

### Eliminar en este archivo

Eliminar la lectura de `R2`, `MAE` y `RMSE` y cualquier comparación basada en ellas.

## 11. `ml_dice_game/modeling/tune_best_model.py`

### Como debe quedar

Reemplazar el contenido del archivo por una version equivalente a la siguiente. Este archivo es responsable exclusivamente de cargar el modelo base seleccionado, ejecutar el tuning, evaluar base/tuneado, guardar el modelo final y generar sus reportes.

```python
import json
from pathlib import Path

import pandas as pd
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from ml_dice_game.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    load_params,
)
from ml_dice_game.explain import ShapExplainer
from ml_dice_game.modeling.common import (
    evaluate_model,
    load_model,
    load_train_test,
    make_cv,
    save_json,
    save_model,
    split_xy,
)
from ml_dice_game.plots import EvaluationReporter


app = typer.Typer()


@app.command()
def main(
    selection_path: Path = MODELS_DIR / "selection.json",
    train_path: Path = PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    metrics_path: Path = REPORTS_DIR / "metrics" / "base_vs_tuned.json",
) -> None:
    params = load_params()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_name = selection["selected_model"]

    base_path = Path(selection["model_path"])
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path

    train_frame, test_frame = load_train_test(train_path, test_path)
    X_train, y_train = split_xy(train_frame)
    X_test, y_test = split_xy(test_frame)
    base_model = load_model(base_path)

    if selected_name == "XGBoost":
        estimator = XGBClassifier(
            random_state=params["random_state"],
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        params_key = "xgb_param_distributions"
    else:
        estimator = RandomForestClassifier(
            random_state=params["random_state"],
            n_jobs=-1,
        )
        params_key = "rf_param_distributions"

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params["tune"][params_key],
        n_iter=params["tune"]["n_iter"],
        scoring="roc_auc",
        cv=make_cv(
            params["split"]["cv_n_splits"],
            params["random_state"],
        ),
        random_state=params["random_state"],
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_

    base_metrics = evaluate_model(base_model, X_test, y_test)
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test)

    save_model(tuned_model, X_train.columns.tolist(), model_path)
    save_json(
        {
            "target": "VENTAJA",
            "selected_model": selected_name,
            "selection_metric": selection["selection_metric"],
            "base": base_metrics,
            "tuned": tuned_metrics,
            "best_params": search.best_params_,
        },
        metrics_path,
    )

    comparison = pd.DataFrame(
        {"base": base_metrics, "tuned": tuned_metrics}
    ).T
    EvaluationReporter(save=True).plot_metrics_comparison(
        comparison,
        filename="base_vs_tuned_metrics_comparison.png",
    )
    ShapExplainer(tuned_model, X_test).save_global_importance(
        FIGURES_DIR / "shap_global_importance.png"
    )


if __name__ == "__main__":
    app()
```

### Responsabilidad de cada bloque

- `selection_path`: carga `models/selection.json` y determina si se ajusta XGBoost o Random Forest.
- `load_train_test()` y `split_xy()`: cargan los CSV procesados y separan `VENTAJA` de las features.
- `XGBClassifier(...)`: define el estimador XGBoost de clasificación binaria. `objective` y `eval_metric` se fijan explícitamente aquí para el tuning.
- `params_key`: selecciona el espacio de hiperparametros correspondiente al modelo ganador.
- `make_cv(...)`: devuelve el `StratifiedKFold` definido en `common.py`.
- `scoring="roc_auc"`: indica que `RandomizedSearchCV` debe escoger los hiperparametros con mayor ROC-AUC.
- `evaluate_model(...)`: calcula `Accuracy`, `Precision`, `Recall`, `F1` y `ROC_AUC` para el modelo base y el tuneado.
- `save_model(...)`: publica el modelo tuneado y el archivo de features.
- `save_json(...)`: guarda metricas, objetivo, modelo seleccionado y `best_params`.

La configuracion de `n_iter`, los espacios `xgb_param_distributions`/`rf_param_distributions` y el numero de folds se mantienen en `params.yaml`; no se escriben dentro de este archivo.

### Eliminar en este archivo

Eliminar `RandomForestRegressor`, `XGBRegressor`, `scoring="r2"` y cualquier evaluacion de `R2`, `MAE` o `RMSE`. No mover a este archivo la logica de seleccion entre modelos base: esa responsabilidad permanece en `select_best_model.py`.

## 12. `params.yaml`

### Cambios exactos

Modificar la sección `split`:

```yaml
split:
  test_size: 0.2
  stratify_col: VENTAJA
  cv_n_splits: 5
```

Modificar la sección `train.xgb`:

```yaml
train:
  rf:
    n_estimators: 300
  xgb:
    n_estimators: 300
    objective: binary:logistic
    eval_metric: logloss
```

Estos parametros se modifican exclusivamente en `params.yaml`, no en `ml_dice_game/modeling/train_xgboost.py`. `model_params()` combina la seccion `train.xgb` con los parametros comunes:

```python
def model_params(self) -> dict[str, Any]:
    return {
        **self.params["train"][self.train_params_key],
        "random_state": self.params["random_state"],
        "n_jobs": -1,
    }
```

Por eso `XGBClassifier(**self.model_params())` recibe `objective: binary:logistic` y `eval_metric: logloss` sin duplicarlos en el entrenador.

Modificar tuning:

```yaml
tune:
  n_iter: 50
  scoring: roc_auc
```

Mantener los espacios de hiperparametros que sean validos para `RandomForestClassifier` y `XGBClassifier`.

Cambiar el nombre registrado:

```yaml
mlflow:
  registered_model_name: ml_dice_game_ventaja_classifier
```

### No agregar aqui

No agregar `reg:squarederror`, `rmse`, `r2`, `RandomForestRegressor` ni `XGBRegressor`.

## 13. `ml_dice_game/plots.py`

### Cambios exactos

Modificar las funciones de comparación para trabajar con estas columnas:

```text
Accuracy
Precision
Recall
F1
ROC_AUC
```

Mantener o agregar funciones para:

- matriz de confusión;
- curva ROC;
- `classification_report`;
- comparación de métricas entre modelos base;
- comparación base frente a tuneado.

### Eliminar en este archivo

Eliminar gráficos de `R2`, `MAE`, `RMSE`, reales frente a predichos y residuos si fueron agregados para el enfoque de regresión.

## 14. `ml_dice_game/explain.py`

### Cambios exactos

Mantener SHAP para modelos de árboles de clasificación. La explicación global y local debe referirse a la clase positiva `VENTAJA=1`.

Si la versión de SHAP devuelve una estructura con dimensión de clases, seleccionar la clase positiva. Si devuelve un arreglo bidimensional, usarlo directamente:

```python
if shap_values.values.ndim == 3:
    shap_values = shap_values[..., 1]
```

Actualizar textos que hablen de regresión, valor continuo o contribución a `PUNTAJE`.

### No modificar aqui

No cambiar SHAP por una explicación de regresión. El modelo continúa siendo un clasificador.

## 15. `dvc.yaml`

### Cambios exactos

En `validate_data`, usar el dataset oficial:

```yaml
deps:
  - data/raw/dataset2.csv
```

Agregar `ml_dice_game/config.py` como dependencia de los stages que dependen de `TARGET_COLUMN`.

Mantener los stages existentes, pero verificar que sus outputs de métricas sean los de clasificación. El pipeline debe depender de:

```text
ml_dice_game/modeling/common.py
ml_dice_game/modeling/train_model.py
ml_dice_game/modeling/train_random_forest.py
ml_dice_game/modeling/train_xgboost.py
ml_dice_game/modeling/select_best_model.py
ml_dice_game/modeling/tune_best_model.py
```

### No modificar aqui

No agregar un stage separado de regresión ni outputs con nombres de métricas `R2`, `MAE` o `RMSE`.

## 16. `api/schemas.py`

### Cambios exactos

El contrato de respuesta debe representar clasificación:

```python
class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_source: str
```

Para batch:

```python
class BatchPredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    model_source: str
```

`prediction` es la clase y `probability` es la probabilidad de `VENTAJA=1`.

### No modificar aqui

Mantener la creación dinámica de `PredictRequest` desde `model.features.json`. No agregar un umbral manual si el modelo ya expone `predict()` y `predict_proba()`.

## 17. `api/model_service.py`

### Cambios exactos

Modificar `predict()` para usar:

```python
classes = self.model.predict(df)
probabilities = self.model.predict_proba(df)[:, 1]
```

Devolver ambas listas, por ejemplo:

```python
return {
    "predictions": [int(value) for value in classes],
    "probabilities": [float(value) for value in probabilities],
}
```

Mantener la carga desde MLflow y el fallback local. Cambiar únicamente el nombre del modelo registrado mediante `params.yaml`.

### Eliminar en este archivo

Eliminar cualquier retorno que trate la predicción como valor continuo de regresión o que aplique `prediction >= threshold`.

## 18. `api/main.py`

### Cambios exactos

Modificar `/predict` para construir `PredictResponse` con:

- clase entera de `VENTAJA`;
- probabilidad de `VENTAJA=1`;
- origen del modelo.

Modificar `/predict/batch` para construir `BatchPredictResponse` con las listas correspondientes.

Mantener la conversión del request a `DataFrame` y la carga dinámica de features.

## 19. `tests/test_dataset.py`

### Cambios exactos

Cambiar fixtures para que incluyan `VENTAJA` en lugar de `PUNTAJE`.

Agregar pruebas para:

- columna `VENTAJA` ausente;
- `VENTAJA` con nulos;
- `VENTAJA` con valores distintos de `0` y `1`;
- carta con suma incorrecta;
- tablero fuera de rango;
- reporte con los nuevos conteos.

## 20. `tests/test_features.py`

### Cambios exactos

Cambiar los fixtures de `PUNTAJE` a `VENTAJA` y comprobar:

```python
features, target = split_xy(frame)
assert "VENTAJA" not in features.columns
assert set(target.unique()).issubset({0, 1})
```

Agregar una prueba de que train y test conservan las proporciones de `VENTAJA` y otra para que `get_cv()` devuelva `StratifiedKFold`.

## 21. `tests/test_modeling.py`

### Cambios exactos

Usar `DummyClassifier` para probar el contrato de clasificación:

```python
from sklearn.dummy import DummyClassifier

model = DummyClassifier(strategy="prior").fit(X, y)
metrics = evaluate_model(model, X, y)
assert set(metrics) == {
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
}
```

Agregar pruebas de que los entrenadores construyen `RandomForestClassifier` y `XGBClassifier`.

## 22. `README.md`

### Cambios exactos

Reemplazar:

- `PUNTAJE` por `VENTAJA`;
- regresión por clasificación binaria;
- `RandomForestRegressor` por `RandomForestClassifier`;
- `XGBRegressor` por `XGBClassifier`;
- `R2`, `MAE` y `RMSE` por `Accuracy`, `Precision`, `Recall`, `F1` y `ROC_AUC`;
- salida continua por clase y probabilidad.

Actualizar el diagrama del pipeline, las rutas del dataset y el contrato de `/predict`.

## 23. Artefactos que deben regenerarse

Después de cambiar el código, ejecutar el pipeline para regenerar, no editar manualmente:

```text
models/base/random_forest.pkl
models/base/random_forest.features.json
models/base/xgboost.pkl
models/base/xgboost.features.json
models/model.pkl
models/model.features.json
models/selection.json
reports/metrics/random_forest.json
reports/metrics/xgboost.json
reports/metrics/base_models.json
reports/metrics/base_vs_tuned.json
reports/figures/base_models_metrics_comparison.png
reports/figures/base_vs_tuned_metrics_comparison.png
reports/figures/shap_global_importance.png
```

## 24. Orden de ejecución

1. Confirmar columnas y ruta de `dataset2.csv`.
2. Completar validaciones en `dataset.py`.
3. Confirmar `TARGET_COLUMN` y `stratify_col` en `config.py` y `params.yaml`.
4. Cambiar `features.py` a `StratifiedKFold` y split estratificado por `VENTAJA`.
5. Cambiar `common.py` y `train_model.py` a clasificadores.
6. Cambiar los dos entrenadores a `RandomForestClassifier` y `XGBClassifier`.
7. Actualizar selección y tuning.
8. Actualizar gráficos, SHAP, API, pruebas y README.
9. Ejecutar `dvc repro`.
10. Ejecutar las pruebas automatizadas y validar `/health`, `/predict` y `/predict/batch`.
11. Confirmar que el notebook no cambió y que esta guia no fue agregada a MkDocs.

## 25. Validación final

Desde la raíz del proyecto:

```powershell
dvc status
dvc repro
python -m pytest tests
git diff --check
```

La adaptación está completa cuando:

- `VENTAJA` es el objetivo en todos los stages;
- `VENTAJA` no aparece entre las features;
- ambos modelos son clasificadores;
- se utiliza `StratifiedKFold`;
- la selección usa `ROC_AUC`;
- la API devuelve clase y probabilidad;
- los artefactos fueron regenerados;
- las pruebas pasan;
- el notebook permanece sin cambios;
- la guia permanece fuera de `docs/mkdocs.yml`.
