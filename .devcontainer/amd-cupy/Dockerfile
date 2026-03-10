# There is no RAPIDS image for ROCm so we focus on cupy
FROM cupy/cupy-rocm:latest

# The base image ships a stale ROCm 5.0.1 amdgpu repo that 404s.
# Remove it so apt-get update succeeds.
RUN rm -f /etc/apt/sources.list.d/amdgpu.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio
