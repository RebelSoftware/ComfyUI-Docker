# Use NVIDIA CUDA 12.8 devel image for compilation support
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

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

# System deps including Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
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
    curl \
 && ln -sf /usr/bin/python3.12 /usr/bin/python \
 && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

# Create runtime user/group (handle existing GID/UID gracefully)
RUN (groupadd --gid 1000 appuser 2>/dev/null || true) \
 && (useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser 2>/dev/null || true) \
 && mkdir -p /home/appuser \
 && chown -R 1000:1000 /home/appuser

# Workdir
WORKDIR /app/ComfyUI

# Leverage layer caching: install deps before copying full tree
COPY requirements.txt ./

# Core Python deps (torch CUDA 12.8, ComfyUI reqs), media/NVML libs
# Skip upgrading system pip/setuptools/wheel to avoid Debian package conflicts
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 \
 && python -m pip install triton \
 && python -m pip install -r requirements.txt \
 && python -m pip install imageio-ffmpeg "av>=14.2" nvidia-ml-py

# Copy the application
COPY . .

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && chown -R appuser:appuser /app /home/appuser /entrypoint.sh

EXPOSE 8188

# Start as root so entrypoint can adjust ownership and drop privileges
USER root
ENTRYPOINT ["/entrypoint.sh"]

# Default command - entrypoint will add --use-sage-attention if available
CMD ["python", "main.py", "--listen", "0.0.0.0"]
