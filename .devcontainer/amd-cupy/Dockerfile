FROM rocm/dev-ubuntu-22.04:latest

# Install Python 3.12, pip, and the full ROCm HIP SDK (includes all
# headers CuPy needs: hip, hiprand, hipblas, hipsparse, hipfft, etc.)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git wget \
    rocm-hip-sdk \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.11 get-pip.py

# Set ROCm environment variables for CuPy
ENV ROCM_HOME=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV HIP_PLATFORM=amd
ENV CUPY_INSTALL_USE_HIP=1
ENV CPATH=/opt/rocm/include
ENV LIBRARY_PATH=/opt/rocm/lib
ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV PATH=/opt/rocm/bin:$PATH

# Install CuPy with ROCm/HIP support
RUN pip3.11 install --no-cache-dir cupy

WORKDIR /workspace

RUN pip3.11 install --no-cache-dir --ignore-installed blinker \
    && pip3.11 install --no-cache-dir \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio
