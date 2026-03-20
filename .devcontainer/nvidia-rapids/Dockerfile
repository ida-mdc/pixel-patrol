# RAPIDS provides cupy, cudf, dask-cuda in a single image
FROM nvcr.io/nvidia/rapidsai/base:25.02-cuda12.8-py3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# cupy and dask-cuda come from the base image; add profiling tools
RUN pip install --no-cache-dir \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio
