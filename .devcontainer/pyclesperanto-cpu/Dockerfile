FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl \
        # OpenCL ICD loader + headers
        ocl-icd-libopencl1 opencl-headers ocl-icd-opencl-dev \
        # For Intel/AMD CPU OpenCL fallback (useful on machines without GPU)
        pocl-opencl-icd \
        # clinfo for diagnostics
        clinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio \
    pyclesperanto
