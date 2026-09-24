# syntax=docker/dockerfile:1
FROM python:3.12-slim

# WITH_CUDA=true builds the GPU image: torch from the cu129 index, which also
# brings the cuBLAS/cuDNN user-space libs CTranslate2 needs. CUDA 12.9 runs on
# the k3s host's 550 driver (CUDA 12.4) via minor-version compatibility;
# verified with CTranslate2's 12.8 build on 2026-09-23. It covers sm_89 (prod
# 4070 Super) and sm_120 (the 5090 dev box).
ARG WITH_CUDA=false
ARG TORCH_VERSION=2.13.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# torch first, from the PyTorch index: installed after requirements.txt,
# transformers would drag in PyPI's default (CUDA, multi-GB) torch even into
# the CPU image. Used by the forced aligner (app/align.py).
RUN if [ "$WITH_CUDA" = "true" ]; then IDX=cu129; else IDX=cpu; fi \
    && pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/$IDX" \
         "torch==$TORCH_VERSION"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Harmless when the CUDA wheels aren't installed.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

COPY app ./app

RUN useradd --uid 1000 --create-home subgen \
    && mkdir -p /data /models /media \
    && chown -R subgen:subgen /data /models /srv
USER subgen

# Two glibc malloc arenas instead of 8 per core: large short-lived numpy/torch
# buffers on the worker thread fragmented the heap until RSS hit the 12Gi
# limit after a few hundred jobs (see app/memory.py).
ENV MALLOC_ARENA_MAX=2

ENV STATE_DIR=/data \
    MODEL_DIR=/models \
    MEDIA_DIRS=/media \
    HF_HOME=/models

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"

CMD ["python", "-m", "app.main"]
