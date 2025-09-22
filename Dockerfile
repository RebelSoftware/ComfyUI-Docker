# --------------------------
# Stage 1: build SageAttention 2.2 wheel (Debian trixie + nvcc)
# --------------------------
FROM python:3.12.11-slim-trixie AS sage-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Write explicit Debian sources with contrib/non-free/non-free-firmware, then install CUDA toolkit + build deps
RUN set -eux; \
  printf 'deb http://deb.debian.org/debian trixie main contrib non-free non-free-firmware\n' > /etc/apt/sources.list; \
  printf 'deb http://deb.debian.org/debian trixie-updates main contrib non-free non-free-firmware\n' >> /etc/apt/sources.list; \
  printf 'deb http://security.debian.org/debian-security trixie-security main contrib non-free non-free-firmware\n' >> /etc/apt/sources.list; \
  apt-get update; \
  apt-get install -y --no-install-recommends ca-certificates curl git build-essential cmake nvidia-cuda-toolkit; \
  rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/sage

# Install Torch cu129 in builder (matches runtime)
RUN python -m pip install --upgrade pip setuptools wheel --break-system-packages && \
    python -m pip install torch torchvision torchaudio \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      --break-system-packages

# Shallow clone SageAttention and build a cp312 wheel
RUN git clone --depth 1 https://github.com/thu-ml/SageAttention.git . && \
    python -m pip wheel . --no-deps --no-build-isolation -w /dist

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
    git build-essential cmake libgl1 libglx-mesa0 libglib2.0-0 \
    fonts-dejavu-core fontconfig util-linux \
 && rm -rf /var/lib/apt/lists/*

# Create runtime user/group
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

WORKDIR /app/ComfyUI

# Install core deps (Torch cu129 must match builder)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch torchvision torchaudio \
      --extra-index-url https://download.pytorch.org/whl/cu129 && \
    python -m pip install -r requirements.txt && \
    python -m pip install imageio-ffmpeg "av>=14.2" nvidia-ml-py

# Install the SageAttention wheel built in the builder stage
COPY --from=sage-builder /dist/sageattention-*.whl /tmp/
RUN python -m pip install /tmp/sageattention-*.whl

# Copy the application
COPY . .

# Entrypoint and launch
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && chown -R appuser:appuser /app /home/appuser /entrypoint.sh

EXPOSE 8188
USER root
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py", "--listen", "0.0.0.0", "--use-sage-attention"]
