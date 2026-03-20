FROM intel/oneapi-basekit:2024.2.1-0-devel-ubuntu22.04

RUN apt update && apt install -y python3.11 clinfo \
    opencl-headers ocl-icd-opencl-dev python3.11-dev cmake wget

# Install Intel NEO compute-runtime 24.35 with IGC for Arc GPU support
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

# The oneAPI base image registers a CPU OpenCL runtime via intel64.icd and sets
# OCL_ICD_FILENAMES to it. Remove both so the ICD loader only finds the GPU
# driver (intel.icd installed by the NEO compute-runtime).
RUN rm -f /etc/OpenCL/vendors/intel64.icd
ENV OCL_ICD_FILENAMES=""

# Set python3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.11
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.11 get-pip.py

WORKDIR /workspace

RUN pip3.11 install --no-cache-dir \
    --break-system-packages \
    --ignore-installed \
    --no-binary pyopencl \
    pixel-patrol-base \
    pixel-patrol-image \
    pixel-patrol-loader-bio \
    pyclesperanto
