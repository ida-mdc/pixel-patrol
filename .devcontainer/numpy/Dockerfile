FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio
