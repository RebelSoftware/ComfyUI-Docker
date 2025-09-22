# Use a recent slim base image
FROM python:3.12.11-slim-trixie

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY_AUTO_INSTALL=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps
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
 && rm -rf /var/lib/apt/lists/*

# Create runtime user/group
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Workdir
WORKDIR /app/ComfyUI

# Leverage layer caching: install deps before copying full tree
COPY requirements.txt ./

# Core Python deps (torch CUDA 12.9, ComfyUI reqs), media/NVML libs
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 \
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
CMD ["python", "main.py", "--listen", "0.0.0.0", "--use-sage-attention"]
