# --------------------------
# Stage 1: build SageAttention 2.2 wheel from source
# --------------------------
FROM python:3.12.11-slim-trixie AS sage-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Build deps + CUDA toolkit (nvcc) from Debian repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake \
    nvidia-cuda-toolkit nvidia-cuda-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/sage

# Match Torch in final image (cu129) before building extension
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129

# Shallow clone latest SageAttention and build a wheel
# (compiles 2.2/2++ from source at repo tip)
RUN git clone --depth 1 https://github.com/thu-ml/SageAttention.git . \
 && python -m pip wheel . --no-deps --no-build-isolation -w /dist

# --------------------------
# Stage 2: your runtime image
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

# Copy requirements first for layer caching
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
