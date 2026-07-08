# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set WITH_CUDA=true to bake in the cuBLAS/cuDNN user-space libs that
# faster-whisper needs for GPU inference (adds ~1.5GB to the image).
ARG WITH_CUDA=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_CUDA" = "true" ]; then \
         pip install --no-cache-dir "nvidia-cublas-cu12" "nvidia-cudnn-cu12>=9,<10"; \
       fi

# Harmless when the CUDA wheels aren't installed.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

COPY app ./app

RUN useradd --uid 1000 --create-home subgen \
    && mkdir -p /data /models /media \
    && chown -R subgen:subgen /data /models /srv
USER subgen

ENV STATE_DIR=/data \
    MODEL_DIR=/models \
    MEDIA_DIRS=/media \
    HF_HOME=/models

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["python", "-m", "app.main"]
