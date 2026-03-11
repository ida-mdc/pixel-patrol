FROM intel/oneapi-basekit:2025.0.0-0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    clinfo git wget \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Install Intel NEO compute-runtime with IGC for Arc GPU / Data Center GPU support
RUN mkdir -p /tmp/neo && cd /tmp/neo && \
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.20/intel-igc-core_1.0.17537.20_amd64.deb && \
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.20/intel-igc-opencl_1.0.17537.20_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/libigdgmm12_22.5.0_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/intel-opencl-icd_24.35.30872.22_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/intel-level-zero-gpu_1.3.30872.22_amd64.deb && \
    dpkg -i libigdgmm12_22.5.0_amd64.deb intel-igc-core_1.0.17537.20_amd64.deb \
            intel-igc-opencl_1.0.17537.20_amd64.deb intel-opencl-icd_24.35.30872.22_amd64.deb \
            intel-level-zero-gpu_1.3.30872.22_amd64.deb && \
    rm -rf /tmp/neo

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Install JAX + Intel Extension for OpenXLA (PJRT plugin for Intel GPUs)
# Version pinning is strict — jax/jaxlib must match the extension version.
RUN pip install --no-cache-dir \
    jax==0.4.38 \
    jaxlib==0.4.38 \
    intel-extension-for-openxla==0.6.0

WORKDIR /workspace

RUN pip install --no-cache-dir --ignore-installed blinker \
    && pip install --no-cache-dir \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio
