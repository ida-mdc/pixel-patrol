FROM rocm/dev-ubuntu-24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    --break-system-packages \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio \
    pyclesperanto
