FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

COPY api ./api
COPY ml_dice_game ./ml_dice_game
COPY models ./models
COPY params.yaml ./params.yaml

RUN useradd --create-home --shell /usr/sbin/nologin appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]