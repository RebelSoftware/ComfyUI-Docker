# Use a recent slim base image
FROM python:3.12.11-slim-trixie

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY_AUTO_INSTALL=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    EXT_PARALLEL=4 \
    NVCC_APPEND_FLAGS="--threads 8" \
    MAX_JOBS=32 \
    SAGE_ATTENTION_AVAILABLE=0 \
    COMFYUI_PATH=/app/ComfyUI \
    COMFYUI_MODEL_PATH=/app/ComfyUI/models \
    COMFYUI_MODELS_PATH=/app/ComfyUI/models

# Enable non-free repositories and install system deps + CUDA toolkit
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    fonts-dejavu-core \
    fontconfig \
    util-linux \
    wget \
    gnupg2 \
    ca-certificates \
    ninja-build \
    patch \
    pkg-config \
    libcairo2-dev \
 && echo "deb http://deb.debian.org/debian trixie main contrib non-free non-free-firmware" > /etc/apt/sources.list.d/non-free.list \
 && wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-9 \
    cuda-cudart-dev-12-9 \
    libcusparse-dev-12-9 \
    libcublas-dev-12-9 \
    libcurand-dev-12-9 \
    libcusolver-dev-12-9 \
    libcufft-dev-12-9 \
    nvidia-smi \
 && rm -rf /var/lib/apt/lists/* \
 && rm cuda-keyring_1.1-1_all.deb

# Install uv (latest)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Patch CUDA math_functions.h for glibc 2.41 compatibility
RUN sed -i 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x) noexcept (true);/' /usr/local/cuda-12.9/include/crt/math_functions.h && \
    sed -i 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x) noexcept (true);/' /usr/local/cuda-12.9/include/crt/math_functions.h && \
    sed -i 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x) noexcept (true);/' /usr/local/cuda-12.9/include/crt/math_functions.h && \
    sed -i 's/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x);/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x) noexcept (true);/' /usr/local/cuda-12.9/include/crt/math_functions.h

# Set CUDA paths for entrypoint compilation
ENV CUDA_HOME=/usr/local/cuda-12.9 \
    PATH=/usr/local/cuda-12.9/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64

# Create symlink for compatibility
RUN ln -sf /usr/local/cuda-12.9 /usr/local/cuda

# Create runtime user/group
RUN set -e; \
    if getent group 1000 >/dev/null 2>&1; then \
        EXISTING_GROUP=$(getent group 1000 | cut -d: -f1); \
        if [ "$EXISTING_GROUP" != "appuser" ]; then groupadd appuser; fi; \
    else \
        groupadd --gid 1000 appuser; \
    fi; \
    if getent passwd 1000 >/dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd 1000 | cut -d: -f1); \
        if [ "$EXISTING_USER" != "appuser" ]; then useradd --gid appuser --create-home --shell /bin/bash appuser; fi; \
    else \
        useradd --uid 1000 --gid appuser --create-home --shell /bin/bash appuser; \
    fi; \
    mkdir -p /home/appuser; \
    chown appuser:appuser /home/appuser

# Workdir
WORKDIR /app/ComfyUI

# Copy requirements with optional handling
COPY requirements.txt* ./

# Core Python deps (torch CUDA 12.9, pin Triton, plus common deps)
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 \
 && python -m pip install "triton==3.4.0" \
 && python -m pip install --prefer-binary cupy-cuda12x \
 && if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi \
 && python -m pip install imageio-ffmpeg "av>=14.2" nvidia-ml-py onnxruntime-gpu \
 && python -m pip install toml GitPython

# Copy the application
COPY . .

# Pre-bake ComfyUI-Manager to reduce runtime work (entrypoint will still update)
RUN mkdir -p /app/ComfyUI/custom_nodes \
 && if [ ! -d "/app/ComfyUI/custom_nodes/ComfyUI-Manager" ]; then \
      git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git /app/ComfyUI/custom_nodes/ComfyUI-Manager || true; \
    fi

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && chown appuser:appuser /app /home/appuser /entrypoint.sh

EXPOSE 8188

# Start as root so entrypoint can adjust ownership and drop privileges
USER root
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py", "--listen", "0.0.0.0"]
