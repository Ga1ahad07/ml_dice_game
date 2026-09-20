# ml_dice_game

Modelo de clasificación binaria para estimar si jugar una carta (dado un estado específico del tablero) dejará al jugador en una posición ventajosa o no, dentro de un juego de mesa en proceso de desarrollo.

El proyecto cubre el ciclo completo de un producto de ML:
- Validación y versionado de datos con **DVC**
- Entrenamiento y comparación de modelos (**Random Forest** y **XGBoost**) con búsqueda de hiperparámetros
- Tracking de experimentos con **MLflow**, explicabilidad con **SHAP**
- Despliegue del modelo final como una **API REST** construida con **FastAPI**, empaquetada en **Docker**.

## Tabla de contenidos

- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Datos](#datos)
- [Pipeline de ML (DVC)](#pipeline-de-ml-dvc)
- [Comandos disponibles (Makefile)](#comandos-disponibles-makefile)
- [Seguimiento de experimentos (MLflow)](#seguimiento-de-experimentos-mlflow)
- [Configuración de parámetros](#configuración-de-parámetros)
- [API de inferencia](#api-de-inferencia)
- [Docker](#docker)
- [Tests](#tests)
- [Autor](#autor)

## Estructura del proyecto

```
ml_dice_game/
├── api/                    # API de inferencia (FastAPI)
│   ├── main.py              # Endpoints: /health, /predict, /predict/batch
│   ├── model_service.py     # Carga y ejecución del modelo final
│   └── schemas.py           # Esquemas Pydantic de entrada/salida
├── data/
│   ├── raw/                 # Datos crudos (dataset2.csv, versionado con DVC)
│   ├── interim/              # Datos intermedios (reporte de validación)
│   ├── processed/            # train.csv / test.csv
│   └── external/             # Datos de fuentes externas
├── ml_dice_game/            # Código fuente del paquete
│   ├── config.py             # Rutas, constantes y carga de params.yaml
│   ├── dataset.py            # Carga y validación de reglas de negocio
│   ├── features.py           # División train/test estratificada
│   ├── explain.py            # Explicabilidad con SHAP
│   ├── pipeline.py           # CLI con las etapas del pipeline (Typer)
│   └── modeling/
│       ├── common.py           # Entrenamiento con CV, métricas, gráficos
│       ├── train_random_forest.py
│       ├── train_xgboost.py
│       ├── train_model.py
│       ├── select_best_model.py
│       ├── tune_best_model.py
│       ├── evaluate_test.py
│       ├── tracking.py         # Integración con MLflow
│       └── predict.py          # Carga de modelo final para inferencia
├── models/                   # Modelos entrenados (base y final)
├── reports/                  # Métricas y figuras generadas
│   └── figures/
├── docs/                      # Documentación con MkDocs
├── notebooks/                 # Notebooks de exploración
├── tests/                     # Pruebas unitarias y de contrato (pytest)
├── dvc.yaml / dvc.lock         # Definición y estado del pipeline DVC
├── params.yaml                 # Hiperparámetros y configuración del pipeline
├── Dockerfile / docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Requisitos

- Python 3.11
- [DVC](https://dvc.org/) (con soporte para Google Drive como remote)
- Docker y Docker Compose (opcional, para despliegue)
- Acceso al remote de datos configurado en `.dvc/config`

## Instalación

Clonar el repositorio y crear un entorno virtual:

```bash
git clone <url-del-repositorio>
cd ml_dice_game

python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate

make requirements
```

Esto instala todas las dependencias listadas en `requirements.txt` (scikit-learn, XGBoost, MLflow, DVC, FastAPI, SHAP, entre otras).

## Datos

Los datos crudos (`data/raw/dataset2.csv`) están versionados con DVC y almacenados en un remote de Google Drive:

```bash
dvc pull
```

El dataset representa jugadas de un juego de cartas/dados con columnas de cartas (`C1A`…`C3C`), tablero (`T1A`…`T3C`), ronda (`RONDA`), turno (`TURNO`) y la variable objetivo `VENTAJA` (0/1), que indica si la jugada dejó al jugador en posición ventajosa.

## Pipeline de ML (DVC)

El pipeline está definido en `dvc.yaml` y se reproduce de extremo a extremo con:

```bash
make pipeline
# equivalente a: dvc repro
```

Etapas del pipeline:

| Etapa      | Descripción                                                                 | Salidas principales |
|------------|------------------------------------------------------------------------------|----------------------|
| `validate` | Valida el dataset crudo contra las reglas de negocio del juego               | `data/interim/validation_report.json` |
| `split`    | Divide los datos en train/test de forma estratificada                       | `data/processed/train.csv`, `data/processed/test.csv` |
| `train`    | Entrena Random Forest y XGBoost con validación cruzada, registra en MLflow y selecciona el mejor modelo por ROC-AUC | `models/base/*.pkl`, `reports/metrics/cv.json`, `reports/metrics/selected_model.json`, `reports/figures/oof/` |
| `tune`     | Ajusta hiperparámetros del modelo seleccionado (`RandomizedSearchCV`) y evalúa en test | `models/final/model.pkl`, `models/final/features.json`, `reports/metrics/test.json`, `reports/figures/test/` |
| `explain`  | Genera explicaciones SHAP (importancia global y beeswarm) del modelo final   | `reports/figures/shap/` |

Cada etapa puede ejecutarse individualmente vía la CLI del pipeline (basada en Typer):

```bash
python -m ml_dice_game.pipeline validate
python -m ml_dice_game.pipeline split
python -m ml_dice_game.pipeline train
python -m ml_dice_game.pipeline tune
python -m ml_dice_game.pipeline explain
```

## Comandos disponibles (Makefile)

```bash
make help              # Lista todos los comandos disponibles
make requirements      # Instala dependencias de Python
make pipeline          # Reproduce el pipeline completo con DVC
make mlflow-ui         # Levanta la UI de MLflow (tracking local en ./mlruns)
make serve             # Levanta la API de inferencia en modo desarrollo (localhost:8000)
make docker-build      # Construye las imágenes Docker
make docker-up         # Levanta la API y MLflow con Docker Compose
make docker-down       # Detiene los servicios Docker
make docker-ps         # Muestra el estado de los servicios Docker
make docker-logs       # Muestra los logs de los servicios Docker
make docker-restart    # Reinicia los servicios Docker
make promote VERSION=N # Promueve a "Production" la versión N del modelo registrado en MLflow
make lint              # Verifica estilo de código con ruff
make format            # Formatea el código con ruff
make test              # Ejecuta la suite de tests con pytest
make clean             # Elimina archivos compilados de Python
```

## Seguimiento de experimentos (MLflow)

Todos los entrenamientos (CV, tuning y explicabilidad) se registran automáticamente en MLflow mediante `ExperimentTracker` (`ml_dice_game/modeling/tracking.py`), incluyendo métricas, parámetros, artefactos (matrices de confusión, curvas ROC, gráficos SHAP) y el modelo entrenado.

Para explorar los experimentos localmente:

```bash
make mlflow-ui
```

En entornos con Docker, MLflow corre como un servicio independiente (ver [Docker](#docker)) accesible en `http://localhost:5000`.

El experimento por defecto se llama `ml_dice_game` y el modelo final se registra bajo el nombre `ml_dice_game_ventaja_classifier` (configurable en `params.yaml`).

## Configuración de parámetros

Todo el comportamiento del pipeline (semilla aleatoria, tamaño de test, hiperparámetros base y espacios de búsqueda para el tuning, nombre del experimento MLflow) se controla desde `params.yaml`, sin necesidad de modificar código:

```yaml
random_state: 42

split:
  test_size: 0.2
  stratify_col: VENTAJA
  cv_n_splits: 5

train:
  rf:
    n_estimators: 300
  xgb:
    n_estimators: 300
    objective: "binary:logistic"
    eval_metric: "logloss"

tune:
  n_iter: 50
  scoring: roc_auc
  # ... espacios de búsqueda para RandomForest y XGBoost

mlflow:
  experiment_name: ml_dice_game
  registered_model_name: ml_dice_game_ventaja_classifier
```

## API de inferencia

La API se construye con FastAPI (`api/main.py`) y sirve el modelo final generado por la etapa `tune` del pipeline (`models/final/model.pkl` y `models/final/features.json`).

Ejecutar en local:

```bash
make serve
# equivalente a: uvicorn api.main:app --reload --port 8000
```

### Endpoints

| Método | Ruta              | Descripción                                    |
|--------|-------------------|-------------------------------------------------|
| GET    | `/health`         | Estado del servicio y disponibilidad del modelo |
| POST   | `/predict`        | Predicción para una sola jugada                 |
| POST   | `/predict/batch`  | Predicción para un lote de jugadas              |

Ejemplo de solicitud a `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "C1A": 1, "C1B": 0, "C1C": 2, "C2A": 0, "C2B": 3, "C2C": 0,
    "C3A": 1, "C3B": 0, "C3C": 1,
    "T1A": 0, "T1B": 1, "T1C": -1, "T2A": 2, "T2B": 0, "T2C": 0,
    "T3A": -2, "T3B": 1, "T3C": 0,
    "RONDA": 3, "TURNO": 1
  }'
```

Respuesta:

```json
{
  "prediction": 1,
  "probability": 0.81
}
```

La ruta o el nombre de los artefactos del modelo pueden sobreescribirse con las variables de entorno `ML_DICE_MODEL_PATH` y `ML_DICE_FEATURES_PATH`.

## Docker

El proyecto incluye una configuración de Docker Compose con dos servicios:

- **api**: la API de inferencia (FastAPI + Uvicorn), expuesta en el puerto `8000`.
- **mlflow**: servidor de tracking de MLflow con backend SQLite, expuesto en el puerto `5000`.

```bash
make docker-build   # Construye la imagen de la API
make docker-up       # Levanta ambos servicios
make docker-logs     # Sigue los logs
make docker-down     # Detiene los servicios
```

La API espera encontrar el modelo final ya generado en `models/` (producido por `dvc repro` o `make pipeline`) antes de construir la imagen.

## Tests

El proyecto incluye pruebas unitarias y de contrato con `pytest`:

- `tests/test_api.py`: prueba los endpoints de la API con un modelo dummy.
- `tests/test_dvc_contract.py`: valida que el pipeline DVC mantenga las etapas y salidas esperadas.
- `tests/test_modeling_contract.py`: valida las convenciones de nombres de métricas y la lógica de selección del mejor modelo.

```bash
make test
```

También se puede verificar y formatear el estilo de código con `ruff`:

```bash
make lint
make format
```

## Autor

**Garcia Andrade Alex Rafael**