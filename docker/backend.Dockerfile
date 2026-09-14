# Sentinel backend.
# CPU inference by default; see docker-compose.gpu.yml for CUDA.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgl1 / libglib2.0-0 are required by OpenCV even headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend /app/backend
COPY scripts /app/scripts

RUN useradd --create-home --uid 10001 sentinel \
    && mkdir -p /app/storage /app/models /app/data \
    && chown -R sentinel:sentinel /app
USER sentinel

WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/system/health || exit 1

# One worker on purpose: capture threads, the alarm state machine and the
# recognition index are per-process singletons.
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
