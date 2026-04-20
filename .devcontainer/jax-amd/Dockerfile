# JAX with AMD GPU support via ROCm
FROM rocm/dev-ubuntu-24.04:6.2.4-complete

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks system Python as externally-managed (PEP 668).
# In a container there is nothing to protect, so remove the guard.
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

WORKDIR /workspace

# JAX with ROCm support + benchmark dependencies
RUN pip install --no-cache-dir \
    "jax[rocm]" \
    numpy \
    dask \
    matplotlib \
    line_profiler \
    memory_profiler \
    psutil \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio
