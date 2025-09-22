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
    SAGE_ATTENTION_AVAILABLE=0

# System deps + minimal CUDA toolkit for building
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
 && wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-9 \
    cuda-cudart-dev-12-9 \
    nvidia-utils-545 \
 && rm -rf /var/lib/apt/lists/* \
 && rm cuda-keyring_1.1-1_all.deb

# Set CUDA paths for entrypoint compilation
ENV CUDA_HOME=/usr/local/cuda-12.9 \
    PATH=/usr/local/cuda-12.9/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}

# Create symlink for compatibility
RUN ln -sf /usr/local/cuda-12.9 /usr/local/cuda

# Create runtime user/group (fix the original issue)
RUN set -e; \
    if getent group 1000 >/dev/null 2>&1; then \
        EXISTING_GROUP=$(getent group 1000 | cut -d: -f1); \
        echo "GID 1000 exists as group: $EXISTING_GROUP"; \
        if [ "$EXISTING_GROUP" != "appuser" ]; then \
            groupadd appuser; \
        fi; \
    else \
        groupadd --gid 1000 appuser; \
    fi; \
    if getent passwd 1000 >/dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd 1000 | cut -d: -f1); \
        echo "UID 1000 exists as user: $EXISTING_USER"; \
        if [ "$EXISTING_USER" != "appuser" ]; then \
            useradd --gid appuser --create-home --shell /bin/bash appuser; \
        fi; \
    else \
        useradd --uid 1000 --gid appuser --create-home --shell /bin/bash appuser; \
    fi; \
    mkdir -p /home/appuser; \
    chown appuser:appuser /home/appuser

# Workdir
WORKDIR /app/ComfyUI

# Leverage layer caching: install deps before copying full tree
COPY requirements.txt* ./

# Core Python deps (torch CUDA 12.9, ComfyUI reqs), media/NVML libs
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 \
 && python -m pip install triton \
 && if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi \
 && python -m pip install imageio-ffmpeg "av>=14.2" nvidia-ml-py

# Copy the application
COPY . .

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && chown appuser:appuser /app /home/appuser /entrypoint.sh

EXPOSE 8188

# Start as root so entrypoint can adjust ownership and drop privileges
USER root
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py", "--listen", "0.0.0.0"]
