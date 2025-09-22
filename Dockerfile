# ------------------------------------------------------------------------------
# Builder: CUDA devel image to compile SageAttention from source (Ubuntu 24.04)
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
# Configurable ref; use "main" or a tag/branch; set this at build time if needed
ARG SAGE_REF=main
# Cache-buster to force re-resolving the latest commit
ARG SAGE_FORCE_REFRESH=0

# System deps for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git curl \
    build-essential cmake ninja-build pkg-config \
    python3 python3-pip python3-venv python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Ensure recent pip toolchain
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python3 -m pip install --upgrade pip setuptools wheel

# Python deps: Torch (CUDA 12.9), Triton
RUN python3 -m pip install --upgrade \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 \
 && python3 -m pip install "triton>=3.0.0"

# Resolve latest commit for the desired ref and build a wheel for SageAttention 2.x
RUN set -eux; \
    echo "CACHE_BUSTER=${SAGE_FORCE_REFRESH}"; \
    SAGE_SHA="$(git ls-remote https://github.com/thu-ml/SageAttention.git "${SAGE_REF}" | awk '{print $1}' | head -n1)"; \
    echo "Resolved SageAttention ${SAGE_REF} -> ${SAGE_SHA}"; \
    git clone https://github.com/thu-ml/SageAttention.git /build/SageAttention; \
    cd /build/SageAttention; \
    git checkout "${SAGE_SHA}"; \
    export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32; \
    python3 -m pip wheel --no-build-isolation --no-deps -w /wheels .

# ------------------------------------------------------------------------------
# Final image: CUDA runtime (Ubuntu 24.04) + Python 3 + ComfyUI + SageAttention
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    COMFY_AUTO_INSTALL=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git curl \
    build-essential cmake \
    libgl1 libglx-mesa0 libglib2.0-0 fonts-dejavu-core fontconfig util-linux \
    python3 python3-pip python3-venv python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Convenience: `python`/`pip` aliases
RUN ln -s /usr/bin/python3 /usr/local/bin/python && ln -s /usr/bin/pip3 /usr/local/bin/pip || true

# Create runtime user/group
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Workdir
WORKDIR /app/ComfyUI

# Copy requirements early to leverage caching
COPY requirements.txt ./

# Core Python deps (Torch CUDA 12.9, app reqs), media/NVML libs, Triton
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 \
 && python -m pip install -r requirements.txt \
 && python -m pip install imageio-ffmpeg "av>=14.2" nvidia-ml-py \
 && python -m pip install "triton>=3.0.0"

# Install SageAttention wheel built in the builder
COPY --from=builder /wheels /tmp/wheels
RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy the application
COPY . .

# Entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown -R appuser:appuser /app /home/appuser /entrypoint.sh

EXPOSE 8188

# Start as root so entrypoint can adjust ownership and drop privileges
USER root
ENTRYPOINT ["/entrypoint.sh"]

# Enable SageAttention by default; override with USE_SAGE_ATTENTION=0 if desired
CMD ["python", "main.py", "--listen", "0.0.0.0", "--use-sage-attention"]
