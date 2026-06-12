# ─── Variables ───────────────────────────────────────────────────────────────
IMAGE_NAME     ?= comfyui-docker
VERSION         = $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' comfyui_version.py)
TAG            ?= v$(VERSION)
DOCKERFILE     ?= Dockerfile
CONTEXT        ?= .
DOCKER_ARGS    ?=
RUN_NAME       ?= comfyui
PORT           ?= 8188
GPU_FLAGS      ?= --gpus all
VOLUMES        ?= -v "$(PWD)/models:/app/ComfyUI/models" \
                  -v "$(PWD)/output:/app/ComfyUI/output" \
                  -v "$(PWD)/custom_nodes:/app/ComfyUI/custom_nodes" \
                  -v "$(PWD)/user:/app/ComfyUI/user"

# Detect if we're on a color-supporting terminal
bold := $(shell tput bold 2>/dev/null || echo "")
reset := $(shell tput sgr0 2>/dev/null || echo "")

# ─── Targets ─────────────────────────────────────────────────────────────────
.PHONY: help build build-nc run shell logs stop push tag info

help: ## Show this help
	@echo "$(bold)Usage:$(reset) make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(bold)%-16s$(reset) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(bold)Current settings:$(reset)"
	@echo "  IMAGE_NAME  = $(IMAGE_NAME)"
	@echo "  VERSION     = $(VERSION)"
	@echo "  TAG         = $(TAG)"
	@echo "  Full image  = $(IMAGE_NAME):$(TAG)"

info: ## Show the resolved image name and tag
	@echo "IMAGE_NAME=$(IMAGE_NAME)"
	@echo "VERSION=$(VERSION)"
	@echo "TAG=$(TAG)"
	@echo "FULL_IMAGE=$(IMAGE_NAME):$(TAG)"

build: ## Build the Docker image (tags with version and latest)
	docker build $(DOCKER_ARGS) \
		-t $(IMAGE_NAME):$(TAG) \
		-t $(IMAGE_NAME):latest \
		-f $(DOCKERFILE) \
		$(CONTEXT)
	@echo ""
	@echo "  Built:  $(IMAGE_NAME):$(TAG)"
	@echo "  Also:   $(IMAGE_NAME):latest"

build-nc: ## Build with no cache
	$(MAKE) build DOCKER_ARGS="--no-cache"

build-progress: ## Build with plain progress output (useful in CI / non-TTY)
	$(MAKE) build DOCKER_ARGS="--progress=plain"

build-nc-progress: ## Build with no cache + plain progress
	$(MAKE) build DOCKER_ARGS="--no-cache --progress=plain"

run: ## Run the container in detached mode
	docker run -d \
		--name $(RUN_NAME) \
		$(GPU_FLAGS) \
		-p $(PORT):8188 \
		$(VOLUMES) \
		--restart unless-stopped \
		$(DOCKER_ARGS) \
		$(IMAGE_NAME):$(TAG)
	@echo "Container '$(RUN_NAME)' started on port $(PORT)."
	@echo "  Logs:  make logs"
	@echo "  Shell: make shell"
	@echo "  Stop:  make stop"

run-rm: ## Run the container and remove it after exit (foreground)
	docker run --rm -it \
		--name $(RUN_NAME) \
		$(GPU_FLAGS) \
		-p $(PORT):8188 \
		$(VOLUMES) \
		$(DOCKER_ARGS) \
		$(IMAGE_NAME):$(TAG)

shell: ## Start a shell inside a running container
	docker exec -it $(RUN_NAME) /bin/bash || \
		docker exec -it $(RUN_NAME) /bin/sh

logs: ## Tail logs from the running container
	docker logs -f $(RUN_NAME)

stop: ## Stop and remove the running container
	docker stop $(RUN_NAME) 2>/dev/null || true
	docker rm $(RUN_NAME) 2>/dev/null || true
	@echo "Container '$(RUN_NAME)' stopped and removed."

tag: ## Re-tag the local image for a registry (set REGISTRY=...)
	$(eval REG ?= $(REGISTRY))
	@if [ -z "$(REG)" ]; then \
		echo "Usage: make tag REGISTRY=ghcr.io/your-org"; \
		exit 1; \
	fi
	docker tag $(IMAGE_NAME):$(TAG) $(REG)/$(IMAGE_NAME):$(TAG)
	docker tag $(IMAGE_NAME):$(TAG) $(REG)/$(IMAGE_NAME):latest
	@echo "Tagged as:"
	@echo "  $(REG)/$(IMAGE_NAME):$(TAG)"
	@echo "  $(REG)/$(IMAGE_NAME):latest"

push: ## Push tagged images to a registry (set REGISTRY=...)
	$(eval REG ?= $(REGISTRY))
	@if [ -z "$(REG)" ]; then \
		echo "Usage: make push REGISTRY=ghcr.io/your-org"; \
		exit 1; \
	fi
	docker push $(REG)/$(IMAGE_NAME):$(TAG)
	docker push $(REG)/$(IMAGE_NAME):latest

clean: ## Remove all local images for this project
	docker rmi $(IMAGE_NAME):$(TAG) $(IMAGE_NAME):latest 2>/dev/null || true
	@echo "Removed local images for '$(IMAGE_NAME)'."
