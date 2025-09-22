# Use NVIDIA CUDA 12.9 devel image for maximum GPU compatibility (RTX 20-50 series)
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

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

# Create runtime user/group with proper error handling
RUN set -e; \
    # Handle existing GID 1000
    if getent group 1000 >/dev/null 2>&1; then \
        EXISTING_GROUP=$(getent group 1000 | cut -d: -f1); \
        echo "GID 1000 exists as group: $EXISTING_GROUP"; \
        if [ "$EXISTING_GROUP" != "appuser" ]; then \
            groupadd appuser; \
            APP_GID=$(getent group appuser | cut -d: -f3); \
        else \
            APP_GID=1000; \
        fi; \
    else \
        groupadd --gid 1000 appuser; \
        APP_GID=1000; \
    fi; \
    # Handle existing UID 1000
    if getent passwd 1000 >/dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd 1000 | cut -d: -f1); \
        echo "UID 1000 exists as user: $EXISTING_USER"; \
        if [ "$EXISTING_USER" != "appuser" ]; then \
            useradd --gid appuser --create-home --shell /bin/bash appuser; \
        fi; \
    else \
        useradd --uid 1000 --gid appuser --create-home --shell /bin/bash appuser; \
    fi; \
    # Ensure home directory exists with correct ownership
    mkdir -p /home/appuser; \
    chown appuser:appuser /home/appuser; \
    echo "Created user: $(id appuser)"; \
    echo "Created group: $(getent group appuser)"

# Workdir
WORKDIR /app/ComfyUI

# Copy requirements.txt with optional handling
COPY requirements.txt* ./

# Core Python deps (torch CUDA 12.9, ComfyUI reqs), media/NVML libs
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 \
 && python -m pip install triton \
 && if [ -f requirements.txt ]; then \
        echo "Installing from requirements.txt"; \
        python -m pip install -r requirements.txt; \
    else \
        echo "No requirements.txt found, skipping"; \
    fi \
 && python -m pip install imageio-ffmpeg "av>=14.2" nvidia-ml-py

# Copy the application
COPY . .

# Entrypoint with proper ownership
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && chown appuser:appuser /app /home/appuser /entrypoint.sh

EXPOSE 8188

# Start as root so entrypoint can adjust ownership and drop privileges
USER root
ENTRYPOINT ["/entrypoint.sh"]

# Default command - entrypoint will add --use-sage-attention if available
CMD ["python", "main.py", "--listen", "0.0.0.0"]
