# CPU-only, reproducible environment.
#   docker build -t crop-rec .
#   docker build -t crop-rec --build-arg EXTRAS=explain,edge,deep .   # adds PyTorch (large)
FROM python:3.11-slim

ARG EXTRAS=explain,edge

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CROP_REC_DATASET="/app/data/New Final DataSet Without Sulphur.xlsx"

# libgomp is required by xgboost / catboost / scikit-learn on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install ".[${EXTRAS}]"

RUN useradd --create-home appuser && mkdir -p /app/models /app/reports && chown -R appuser /app
USER appuser

ENTRYPOINT ["crop-rec"]
CMD ["--help"]
