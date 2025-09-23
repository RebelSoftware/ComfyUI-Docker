<div align="center">

# ComfyUI-Docker
**An automated Repo for ComfyUI Docker image builds, optimized for NVIDIA GPUs.**

[![][github-stargazers-shield]][github-stargazers-link]
[![][github-release-shield]][github-release-link]
[![][github-license-shield]][github-license-link]

[github-stargazers-shield]: https://img.shields.io/github/stars/clsferguson/ComfyUI-Docker.svg
[github-stargazers-link]: https://github.com/clsferguson/ComfyUI-Docker/stargazers
[github-release-shield]: https://img.shields.io/github/v/release/clsferguson/ComfyUI-Docker?style=flat&sort=semver
[github-release-link]: https://github.com/clsferguson/ComfyUI-Docker/releases
[github-license-shield]: https://img.shields.io/github/license/clsferguson/ComfyUI-Docker.svg
[github-license-link]: https://github.com/clsferguson/ComfyUI-Docker/blob/master/LICENSE

[About](#about) • [Features](#features) • [Getting Started](#getting-started) • [Usage](#usage) • [License](#license)

</div>

---

## About
This image packages upstream [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with CUDA-enabled PyTorch and an entrypoint that can build SageAttention at container startup for modern NVIDIA GPUs.

The base image is python:3.12-slim (Debian trixie) with CUDA 12.9 developer libraries installed via apt and PyTorch installed from the cu129 wheel index.

It syncs with the upstream ComfyUI repository, builds a Docker image on new releases, and pushes it to GitHub Container Registry (GHCR).

I created this repo for myself as a simple way to stay up to date with the latest ComfyUI versions while having an easy-to-use Docker image.

---

## Features
- Daily checks for upstream releases, auto-merges changes, and builds/pushes Docker images.
- CUDA-enabled PyTorch + Triton on Debian trixie with CUDA 12.9 dev libs so custom CUDA builds work at runtime.
- Non-root runtime with PUID/PGID mapping handled by entrypoint for volume permissions.
- ComfyUI-Manager auto-sync on startup; entrypoint scans custom_nodes and installs requirements when COMFY_AUTO_INSTALL=1.
- SageAttention build-on-start with TORCH_CUDA_ARCH_LIST tuned to detected GPUs; enabling is opt-in at runtime via FORCE_SAGE_ATTENTION=1.

---

## Getting Started
- Install NVIDIA Container Toolkit on the host, then use docker run --gpus all or Compose GPU reservations to pass GPUs through.
- Expose the ComfyUI server on port 8188 (default) and map volumes for models, inputs, outputs, and custom_nodes.

### Pulling the Image
The latest image is available on GHCR:

```bash
docker pull ghcr.io/clsferguson/comfyui-docker:latest
```

For a specific version (synced with upstream tags, starting at 0.3.59):
```bash
docker pull ghcr.io/clsferguson/comfyui-docker:vX.Y.Z
```

### Docker Compose
For easier management, use this `docker-compose.yml`:

```yaml
services:
  comfyui:
    image: ghcr.io/clsferguson/comfyui-docker:latest
    container_name: ComfyUI
    runtime: nvidia
    restart: unless-stopped
    ports:
      - 8188:8188
    environment:
      - TZ=America/Edmonton
      - PUID=1000
      - GUID=1000
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
- Open http://localhost:8188 after the container is up; change the external port via -p HOST:8188 or the internal port with ComfyUI --port/--listen.
- To target specific GPUs, use Docker’s GPU device selections or Compose device_ids in reservations.

### SageAttention
- The entrypoint builds and caches SageAttention on startup when GPUs are detected; runtime activation is controlled by FORCE_SAGE_ATTENTION=1.
- If the SageAttention import test fails, the entrypoint logs a warning and starts ComfyUI without --use-sage-attention even if FORCE_SAGE_ATTENTION=1.
- To enable: set FORCE_SAGE_ATTENTION=1 and restart; to disable, omit or set to 0.

### Environment Variables
- PUID/PGID: map container user to host UID/GID for volume write access.
- COMFY_AUTO_INSTALL=1: auto-install Python requirements from custom_nodes on startup.
- FORCE_SAGE_ATTENTION=0|1: if 1 and the module import test passes, the entrypoint adds --use-sage-attention.

---

## License
Distributed under the MIT License (same as upstream ComfyUI). See [LICENSE](LICENSE) for more information.

---

## Contact
- **Creator**: clsferguson - [GitHub](https://github.com/clsferguson)
- **Project Link**: https://github.com/clsferguson/ComfyUI-Docker

<p align="center">
  <i>Built with ❤️ for easy AI workflows.</i>
</p>
