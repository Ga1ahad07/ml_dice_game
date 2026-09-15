# ml_dice_game

Modelo de regresion no lineal para estimar `PUNTAJE` a partir del estado del
tablero, la ronda, el turno y la carta jugada en un juego de mesa.

El proyecto compara `RandomForestRegressor` y `XGBRegressor`, selecciona el
mejor modelo por `R2` en test, ajusta sus hiperparametros y publica el modelo
final para consumo local o mediante API.

## Flujo del proyecto

```text
data/raw/dataset.csv
        |
        v
validate_data -> data/interim/validation_report.json
        |
        v
split_data -> data/processed/train.csv + test.csv
        |
        +--> train_random_forest -> models/base/random_forest.pkl
        |                          reports/metrics/random_forest.json
        |
        +--> train_xgboost ------> models/base/xgboost.pkl
                                   reports/metrics/xgboost.json
        |
        v
select_best_model -> models/selection.json
                     reports/metrics/base_models.json
        |
        v
tune_best_model -> models/model.pkl
                   reports/metrics/base_vs_tuned.json
                   reports/figures/*.png
```

El pipeline se define en `dvc.yaml`. Los stages de Random Forest y XGBoost
heredan el flujo comun de `TrainModel`, definido en
`ml_dice_game/modeling/train_model.py`.

## Requisitos

- Python `3.11`, version declarada por el proyecto.
- Git.
- DVC `3.x`.
- Docker Desktop con Docker Compose v2, para ejecutar la API y MLflow en contenedores.
- Acceso al remoto DVC configurado si se necesita descargar o publicar datos.

Las dependencias Python se encuentran en `requirements.txt`. El remoto Google
Drive usa `dvc-gdrive`; las versiones de `pyOpenSSL` y `cryptography` deben
respetar las restricciones compatibles con PyDrive2.

## Instalacion

Desde la raiz del repositorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En PowerShell, si la politica de ejecucion impide activar el entorno, puede
usarse temporalmente:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Tambien es posible instalar mediante Make:

```text
make requirements
```

## Datos y DVC

El dataset original debe estar en `data/raw/dataset.csv` y esta versionado con
DVC:

```text
dvc pull
dvc push
dvc status
dvc dag
```

Las credenciales del remoto deben permanecer en `.dvc/config.local` o en el
mecanismo seguro elegido para el entorno. No deben agregarse al repositorio.

## Ejecutar el pipeline completo

La forma recomendada de reproducir todos los stages es:

```text
dvc repro
```

Tambien puede usarse `make pipeline`. El pipeline ejecuta estas etapas:

1. `validate_data`: valida faltantes, duplicados y reglas de las cartas.
2. `split_data`: crea `train.csv` y `test.csv`, estratificando por `RONDA`.
3. `train_random_forest`: entrena y evalua el baseline Random Forest.
4. `train_xgboost`: entrena y evalua el baseline XGBoost.
5. `select_best_model`: selecciona el modelo con mayor `R2` en test.
6. `tune_best_model`: ajusta el ganador, genera SHAP y publica el modelo final.

Para ejecutar cada stage manualmente:

```text
python -m ml_dice_game.dataset
python -m ml_dice_game.features
python -m ml_dice_game.modeling.train_random_forest
python -m ml_dice_game.modeling.train_xgboost
python -m ml_dice_game.modeling.select_best_model
python -m ml_dice_game.modeling.tune_best_model
```

## Configuracion

Los parametros del experimento se encuentran en `params.yaml`:

- `random_state`: semilla de reproducibilidad.
- `split`: tamano del test, columna de estratificacion y folds de CV.
- `train.rf`: parametros base de Random Forest.
- `train.xgb`: parametros base de XGBoost.
- `tune`: numero de iteraciones, metrica y espacios de hiperparametros.
- `mlflow`: nombre del experimento y del modelo registrado.

No se deben colocar rutas absolutas ni secretos en `params.yaml`.

## Arquitectura de entrenamiento

`TrainModel` aplica un flujo comun mediante herencia:

```text
TrainModel.run()
  cargar train/test
  separar X e y
  crear estimador especifico
  ejecutar cross-validation
  entrenar con todo train
  evaluar en test
  guardar modelo y metricas
```

Las clases hijas solo definen el estimador y la clave de parametros:

- `RandomForestTrainer` usa `RandomForestRegressor` y `train.rf`.
- `XGBoostTrainer` usa `XGBRegressor` y `train.xgb`.

Los artefactos base se guardan en `models/base/`. La etapa de seleccion genera
`models/selection.json`; la etapa de tuning publica siempre `models/model.pkl`,
que es la ruta usada por el fallback local de la API.

## Metricas y reportes

Cada modelo genera un JSON con `R2`, `MAE`, `RMSE`, medias de validacion cruzada
y los parametros efectivos del estimador. Los reportes principales son:

```text
reports/metrics/random_forest.json
reports/metrics/xgboost.json
reports/metrics/base_models.json
reports/metrics/base_vs_tuned.json
reports/figures/base_models_metrics_comparison.png
reports/figures/base_vs_tuned_metrics_comparison.png
reports/figures/shap_global_importance.png
```

## API de prediccion

La API usa FastAPI y carga el modelo registrado en MLflow en `Production`. Si
no puede acceder al registro, usa `models/model.pkl` como fallback local.

Iniciar el servidor:

```text
uvicorn api.main:app --reload --port 8000
```

O mediante `make serve`. La documentacion interactiva queda disponible en
`http://localhost:8000/docs`.

Endpoints disponibles:

- `GET /health`: estado de la API y origen del modelo.
- `POST /predict`: prediccion individual.
- `POST /predict/batch`: predicciones por lote.

El cuerpo de `/predict` debe incluir todas las features listadas en
`models/model.features.json`. Los esquemas se generan dinamicamente en
`api/schemas.py`.

## Ejecucion con Docker

La configuracion de `docker-compose.yml` levanta dos servicios:

- `api`: imagen local `ml-dice-game-api:local`, expuesta en
        `http://localhost:8000`.
- `mlflow`: servidor MLflow expuesto en `http://localhost:5000`, con un volumen
        Docker nombrado para conservar los artefactos almacenados en `/mlruns`.

Antes de construir la imagen, se recomienda ejecutar `dvc repro` para generar
`models/model.pkl` y los metadatos de features. La imagen copia esos artefactos
y la API los usa como fallback local si no encuentra un modelo en `Production`
en el registro MLflow.

Desde la raiz del repositorio, ejecutar:

```powershell
docker compose build
docker compose up -d
docker compose ps
```

Tambien pueden usarse los comandos equivalentes de Make:

```text
make docker-build
make docker-up
make docker-ps
```

Con los servicios activos, la API queda disponible en `http://localhost:8000`
y su documentacion en `http://localhost:8000/docs`. La interfaz de MLflow queda
disponible en `http://localhost:5000`.

Para consultar los logs, reiniciar o detener los servicios:

```powershell
docker compose logs -f
docker compose restart
docker compose down
```

Los equivalentes de Make son `make docker-logs`, `make docker-restart` y
`make docker-down`. `docker compose down` conserva el volumen `mlflow-data`;
para eliminar tambien los datos persistidos de MLflow, usar
`docker compose down -v`.

La API recibe `MLFLOW_TRACKING_URI=http://mlflow:5000` dentro de la red de
Compose. Por eso el servicio debe referenciarse como `mlflow` desde el
contenedor, aunque desde el equipo anfitrion se acceda mediante
`localhost:5000`.

## MLflow

El tracking local se configura mediante `ExperimentTracker`:

```text
mlflow ui --backend-store-uri ./mlruns
```

O mediante `make mlflow-ui`. Para promover una version registrada:

```text
make promote VERSION=1
```

La promocion requiere que la version indicada exista en el Model Registry.

## Notebook

El notebook principal es `notebooks/entrenamiento_rf_xgboost.ipynb`. Contiene
EDA, distribuciones, correlaciones, comparacion de modelos, tuning y SHAP. Para
ejecuciones reproducibles del pipeline se recomienda usar DVC; el notebook se
usa principalmente para exploracion, visualizacion y analisis.

## Pruebas, formato y limpieza

```text
python -m pytest tests
make test
make lint
make format
make clean
```

## Estructura actual

```text
ml_dice_game/
├── api/
│   ├── main.py                  # Endpoints FastAPI
│   ├── model_service.py         # MLflow y fallback local
│   └── schemas.py               # Esquemas de entrada/salida
├── data/
│   ├── raw/                     # Dataset original versionado por DVC
│   ├── interim/                 # Reportes intermedios
│   └── processed/               # train.csv y test.csv
├── docs/                        # Documentacion del proyecto
├── ml_dice_game/
│   ├── config.py                # Rutas y parametros compartidos
│   ├── dataset.py               # Carga y validacion del dataset
│   ├── features.py              # Split train/test
│   ├── explain.py               # Explicabilidad SHAP
│   ├── plots.py                 # Graficos EDA y evaluacion
│   └── modeling/
│       ├── common.py            # Funciones compartidas de ML
│       ├── predict.py           # Predictor serializado
│       ├── tracking.py          # Integracion MLflow
│       ├── train_model.py       # Clase padre del entrenamiento
│       ├── train_random_forest.py
│       ├── train_xgboost.py
│       ├── select_best_model.py
│       └── tune_best_model.py
├── models/                      # Modelos y metadatos generados
├── notebooks/                   # Exploracion y visualizacion
├── reports/                     # Metricas y figuras
├── tests/                       # Pruebas automatizadas
├── dvc.yaml                     # Pipeline reproducible
├── params.yaml                  # Parametros del experimento
├── Makefile                     # Comandos frecuentes
└── requirements.txt             # Dependencias Python
```

## Estado y reproducibilidad

Los modelos, metricas, figuras y datasets generados no deben editarse
manualmente. Cambiar codigo o parametros y ejecutar `dvc repro` permite que DVC
determine que etapas deben repetirse:

```text
python -m pytest tests
dvc status
git diff --check
```
No subir `.env`, `.dvc/config.local`, client secrets ni tokens al repositorio.
