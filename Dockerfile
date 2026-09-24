FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_PROVIDERS=CPUExecutionProvider \
    OMP_NUM_THREADS=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY backend/requirements.txt ./requirements.txt
# Railway runs CPU workloads; avoid downloading CUDA runtime dependencies.
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY backend/ ./
COPY frontend/public/generated-thumbnails/ /app/frontend/public/generated-thumbnails/

# Resolve LFS pointers and validate model inference before deployment.
RUN python scripts/prepare_models.py \
    && python -c 'from app import app; r = app.test_client().get("/"); assert r.status_code == 200 and r.json["status"] == "ok"'

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
