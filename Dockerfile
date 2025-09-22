# --------------------------
# Stage 1: build SageAttention 2.2 wheel from source with nvcc available
# --------------------------
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04 AS sage-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV=/opt/venv

# Python 3.12 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

# Create a venv to avoid PEP 668 'externally-managed-environment'
RUN python3 -m venv "$VENV"
ENV PATH="$VENV/bin:$PATH"

WORKDIR /tmp/sage

# Install Torch (cu129) in the venv before building the extension so ABIs align
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129

# Shallow clone latest SageAttention and build a cp312 wheel from source
RUN git clone --depth 1 https://github.com/thu-ml/SageAttention.git . \
 && python -m pip wheel . --no-deps --no-build-isolation -w /dist

# --------------------------
# Stage 2: runtime image (slim)
# --------------------------
FROM python:3.12.11-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY_AUTO_INSTALL=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake \
    libgl1 libglx-mesa0 libglib2.0-0 \
    fonts-dejavu-core fontconfig util-linux \
 && rm -rf /var/lib/apt/lists/*

# Create runtime user/group
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

WORKDIR /app/ComfyUI

# Install core deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 \
 && python -m pip install -r requirements.txt \
 && python -m pip install imageio-ffmpeg "av>=14.2" nvidia-ml-py

# Install the SageAttention wheel built in the builder stage
COPY --from=sage-builder /dist/sageattention-*.whl /tmp/
RUN python -m pip install /tmp/sageattention-*.whl

# Copy the application
COPY . .

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && chown -R appuser:appuser /app /home/appuser /entrypoint.sh

EXPOSE 8188
USER root
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py", "--listen", "0.0.0.0", "--use-sage-attention"]
