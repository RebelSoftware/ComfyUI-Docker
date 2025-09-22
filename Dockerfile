# --------------------------
# Stage 1: build SageAttention 2.2 wheel from source with nvcc available
# --------------------------
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS sage-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.12 and build tools (Ubuntu 24.04 ships Python 3.12)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

# Make 'python' point to Python 3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /tmp/sage

# Match runtime Torch (cu129) before building the extension so ABIs align
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129

# Shallow clone latest SageAttention and build a cp312 wheel
RUN git clone --depth 1 https://github.com/thu-ml/SageAttention.git . \
 && python -m pip wheel . --no-deps --no-build-isolation -w /dist

# --------------------------
# Stage 2: runtime image (slim)
# --------------------------
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

# Core Python deps (Torch CUDA 12.9, ComfyUI reqs), media/NVML libs
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 \
 && python -m pip install -r requirements.txt \
 && python -m pip install imageio-ffmpeg "av>=14.2" nvidia-ml-py

# Bring in the SageAttention 2.2 wheel compiled in the builder stage and install it
COPY --from=sage-builder /dist/sageattention-*.whl /tmp/
RUN python -m pip install /tmp/sageattention-*.whl

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
