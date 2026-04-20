# JAX with NVIDIA GPU support
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks system Python as externally-managed (PEP 668).
# In a container there is nothing to protect, so remove the guard.
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

WORKDIR /workspace

# JAX with CUDA 12 support + benchmark dependencies
RUN pip install --no-cache-dir \
    "jax[cuda12]" \
    numpy \
    dask \
    matplotlib \
    line_profiler \
    memory_profiler \
    psutil \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio
