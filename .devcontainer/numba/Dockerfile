FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    numpy \
    dask[complete] \
    numba \
    line_profiler \
    memory_profiler \
    psutil \
    matplotlib
