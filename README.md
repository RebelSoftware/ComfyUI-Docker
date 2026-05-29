<div align="center">

# ComfyUI-Docker
**An automated Repo for ComfyUI Docker image builds, optimized for NVIDIA GPUs.**


[![][github-release-shield]][github-release-link]
[![][github-license-shield]][github-license-link]


[github-release-shield]: https://img.shields.io/github/v/release/clsferguson/ComfyUI-Docker?style=flat&sort=semver
[github-release-link]: https://github.com/RebelSoftware/ComfyUI-Docker/releases
[github-license-shield]: https://img.shields.io/github/license/RebelSoftware/ComfyUI-Docker.svg
[github-license-link]: https://github.com/RebelSoftware/ComfyUI-Docker/blob/master/LICENSE

[About](#about) • [Features](#features) • [Getting Started](#getting-started) • [Usage](#usage) • [License](#license)

</div>

---

## About
This image packages upstream [ComfyUI-Docker](https://github.com/clsferguson/ComfyUI-Docker) with updated nvidia drivers 13.2 at the time of writing it also resolves issues with comfyui-manager being rebuilt on every restart  

The base image is python:3.12-slim (Debian trixie) with CUDA 13.2 developer libraries installed via apt and PyTorch installed from the cu130 wheel index.


---

## Features
- Daily checks for upstream releases, auto-merges changes, and builds/pushes Docker images.
- CUDA-enabled PyTorch + Triton on Debian trixie with CUDA 13.2 dev libs so custom CUDA builds work at runtime.
- Non-root runtime with PUID/PGID mapping handled by entrypoint for volume permissions.
- ComfyUI-Manager auto-sync on startup; entrypoint scans custom_nodes and installs requirements when COMFY_AUTO_INSTALL=1.
- SageAttention build-on-start for compatible NVIDIA GPUs (Turing/SM 7.5+); enabling is opt-in via FORCE_SAGE_ATTENTION=1.

---

## Getting Started
- Install NVIDIA Container Toolkit on the host, then use docker run --gpus all or Compose GPU reservations to pass GPUs through.
- Expose the ComfyUI server on port 8188 (default) and map volumes for models, inputs, outputs, and custom_nodes.


### Docker Compose
For easier management, use this `docker-compose.yml`:

```yaml
services:
  comfyui:
    image: comfyui-cuda13:latest # Use the custom-built image with CUDA 13 support built from https://github.com/RebelSoftware/ComfyUI-Docker
    container_name: ComfyUI
    runtime: nvidia
    restart: unless-stopped
    ports:
      - 8188:8188
    environment:
      - TZ=America/Edmonton
      - PUID=1000
      - PGID=1000
    gpus: all
    volumes:
      - comfyui_data:/app/ComfyUI/user/default
      - comfyui_nodes:/app/ComfyUI/custom_nodes
      - /mnt/comfyui/models:/app/ComfyUI/models
      - /mnt/comfyui/input:/app/ComfyUI/input
      - /mnt/comfyui/output:/app/ComfyUI/output
```

Run with `docker compose up -d`.

---

## Usage
- Open http://localhost:8188 after the container is up; change the external port via -p HOST:8188.
- To target specific GPUs, use Docker's GPU device selections or Compose device_ids in reservations.

### SageAttention
SageAttention is compiled at container startup when a compatible GPU (Turing SM 7.5+) is detected and cached to a volume-mapped directory for subsequent starts. It delivers 2-5x faster attention vs FlashAttention for video and high-res image workflows.

- To enable: set `FORCE_SAGE_ATTENTION=1`. If the build or import fails, ComfyUI starts normally without it.
- The first startup with SageAttention will be slower due to compilation; subsequent starts use the cached build.
- Turing GPUs (RTX 20xx) use the v1.0 branch with Triton 3.2.0; Ampere and newer use the latest release.

### Environment Variables
- PUID/PGID: map container user to host UID/GID for volume write access.
- COMFY_AUTO_INSTALL=1: auto-install Python requirements from custom_nodes on startup (default: 1).
- COMFY_FORCE_INSTALL=1: force reinstall of custom_nodes requirements even after first run.
- FORCE_SAGE_ATTENTION=0|1: compile and enable SageAttention on startup (requires compatible NVIDIA GPU).
- SAGE_MAX_JOBS=N: override the number of parallel compile jobs for SageAttention (default: auto from RAM).
- CM_*: seed ComfyUI-Manager config.ini keys on first start (e.g. CM_SKIP_UPDATE_CHECK=1).

---

## License
Distributed under the MIT License (same as upstream ComfyUI). See [LICENSE](LICENSE) for more information.

---

## Contact
- **Creator**: RebelSoftware - [GitHub](https://github.com/RebelSoftware)
- **Project Link**: https://github.com/RebelSoftware/ComfyUI-Docker

