FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY apps/ml-engine /ml-engine
RUN pip install --no-cache-dir /ml-engine

COPY apps/api/pyproject.toml .
COPY apps/api/api ./api
RUN pip install --no-cache-dir ".[dev]"

RUN mkdir -p /app/storage/gradients /app/storage/uploads /app/storage/checkpoints
