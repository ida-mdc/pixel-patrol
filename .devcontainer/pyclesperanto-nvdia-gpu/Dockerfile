FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 nvidia-opencl-dev clinfo nvidia-opencl-icd-384 wget \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.11 get-pip.py

WORKDIR /workspace

RUN pip3.11 install --no-cache-dir \
    --break-system-packages \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio \
    pyclesperanto
