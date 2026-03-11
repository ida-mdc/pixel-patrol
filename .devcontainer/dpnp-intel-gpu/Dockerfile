FROM intel/python:3.11-xpu-full

ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.11 python3.11-dev python3.11-venv \
#     clinfo git wget \
#     && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 \
#     && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
#     && update-alternatives --set python3 /usr/bin/python3.11 \
#     && rm -rf /var/lib/apt/lists/*

# Host has driver 25.18 which matches 2025.3.1 base image.
# No manual NEO compute-runtime install needed.

# Target Intel GPU only via Level Zero and OpenCL (skip non-Intel GPUs)
# ENV ONEAPI_DEVICE_SELECTOR=level_zero:gpu;opencl:gpu

# RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# Install dpnp from Intel's pip index
# RUN pip install --no-cache-dir \
#     dpnp \
#     --extra-index-url https://software.repos.intel.com/python/pypi

WORKDIR /workspace

RUN pip install --no-cache-dir --ignore-installed blinker \
    && pip install --no-cache-dir \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio
