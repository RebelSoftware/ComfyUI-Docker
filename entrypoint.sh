#!/bin/bash
set -euo pipefail

APP_USER=${APP_USER:-appuser}
APP_GROUP=${APP_GROUP:-appuser}
PUID=${PUID:-1000}
PGID=${PGID:-1000}
BASE_DIR=/app/ComfyUI
CUSTOM_NODES_DIR="$BASE_DIR/custom_nodes"
SAGE_ATTENTION_DIR="$BASE_DIR/.sage_attention"
SAGE_ATTENTION_BUILT_FLAG="$SAGE_ATTENTION_DIR/.built"
PERMISSIONS_SET_FLAG="$BASE_DIR/.permissions_set"
FIRST_RUN_FLAG="$BASE_DIR/.first_run_complete"

# Function to log with timestamp
log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Function to test PyTorch CUDA compatibility
test_pytorch_cuda() {
    python -c "
import torch
import sys
if not torch.cuda.is_available():
    print('[ERROR] PyTorch CUDA not available')
    sys.exit(1)
device_count = torch.cuda.device_count()
print(f'[TEST] PyTorch CUDA available with {device_count} devices')
for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f'[TEST] GPU {i}: {props.name} (Compute {props.major}.{props.minor})')
" 2>/dev/null
}

# Function to detect all GPUs and their generations (best-effort labels)
detect_gpu_generations() {
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "")
    local has_rtx20=false
    local has_rtx30=false
    local has_rtx40=false
    local has_rtx50=false
    local gpu_count=0

    if [ -z "$gpu_info" ]; then
        log "No NVIDIA GPUs detected"
        return 1
    fi

    log "Detecting GPU generations:"
    while IFS= read -r gpu; do
        gpu_count=$((gpu_count + 1))
        log "  GPU $gpu_count: $gpu"
        case "$gpu" in
            *"RTX 20"*|*"2060"*|*"2070"*|*"2080"*|*"2090"*) has_rtx20=true ;;
            *"RTX 30"*|*"3060"*|*"3070"*|*"3080"*|*"3090"*) has_rtx30=true ;;
            *"RTX 40"*|*"4060"*|*"4070"*|*"4080"*|*"4090"*) has_rtx40=true ;;
            *"RTX 50"*|*"5060"*|*"5070"*|*"5080"*|*"5090"*) has_rtx50=true ;;
        esac
    done <<< "$gpu_info"

    export DETECTED_RTX20=$has_rtx20
    export DETECTED_RTX30=$has_rtx30
    export DETECTED_RTX40=$has_rtx40
    export DETECTED_RTX50=$has_rtx50
    export GPU_COUNT=$gpu_count

    log "Detection summary: RTX20=$has_rtx20, RTX30=$has_rtx30, RTX40=$has_rtx40, RTX50=$has_rtx50"

    if test_pytorch_cuda; then
        log "PyTorch CUDA compatibility confirmed"
    else
        log "WARNING: PyTorch CUDA compatibility issues detected"
    fi
}

# Function to determine optimal Sage Attention strategy for mixed GPUs
determine_sage_strategy() {
    local strategy=""

    if [ "${DETECTED_RTX20:-false}" = "true" ]; then
        if [ "${DETECTED_RTX30:-false}" = "true" ] || [ "${DETECTED_RTX40:-false}" = "true" ] || [ "${DETECTED_RTX50:-false}" = "true" ]; then
            strategy="mixed_with_rtx20"
            log "Mixed GPU setup detected with RTX 20 series - using compatibility mode"
        else
            strategy="rtx20_only"
            log "RTX 20 series only detected"
        fi
    elif [ "${DETECTED_RTX50:-false}" = "true" ]; then
        strategy="rtx50_capable"
        log "RTX 50 series detected - using latest optimizations"
    elif [ "${DETECTED_RTX40:-false}" = "true" ] || [ "${DETECTED_RTX30:-false}" = "true" ]; then
        strategy="rtx30_40_optimized"
        log "RTX 30/40 series detected - using standard optimizations"
    else
        strategy="fallback"
        log "Unknown or unsupported GPU configuration - using fallback"
    fi

    export SAGE_STRATEGY=$strategy
}

# Function to install appropriate Triton version based on strategy
install_triton_version() {
    case "$SAGE_STRATEGY" in
        "mixed_with_rtx20"|"rtx20_only")
            log "Installing Triton 3.2.0 for RTX 20 series compatibility"
            python -m pip install --no-cache-dir --user --force-reinstall "triton==3.2.0" || {
                log "WARNING: Failed to install specific Triton version, using default"
                python -m pip install --no-cache-dir --user --force-reinstall triton || true
            }
            ;;
        "rtx50_capable")
            log "Installing latest Triton for RTX 50 series"
            python -m pip install --no-cache-dir --user --force-reinstall triton || \
            python -m pip install --no-cache-dir --user --force-reinstall --pre triton || {
                log "WARNING: Failed to install latest Triton, using stable >=3.2.0"
                python -m pip install --no-cache-dir --user --force-reinstall "triton>=3.2.0" || true
            }
            ;;
        *)
            log "Installing latest stable Triton"
            python -m pip install --no-cache-dir --user --force-reinstall triton || {
                log "WARNING: Triton installation failed, continuing without"
                return 1
            }
            ;;
    esac
}

# Function to compute CUDA arch list from torch
compute_cuda_arch_list() {
    python - <<'PY' 2>/dev/null
import torch
archs = set()
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        archs.add(f"{p.major}.{p.minor}")
print(";".join(sorted(archs)))
PY
}

# Function to build Sage Attention with architecture-specific optimizations
build_sage_attention_mixed() {
    log "Building Sage Attention for current GPU environment..."

    mkdir -p "$SAGE_ATTENTION_DIR"
    cd "$SAGE_ATTENTION_DIR"

    local cuda_arch_list
    cuda_arch_list="$(compute_cuda_arch_list || true)"
    if [ -n "${cuda_arch_list:-}" ]; then
        export TORCH_CUDA_ARCH_LIST="$cuda_arch_list"
        log "Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    else
        log "Could not infer TORCH_CUDA_ARCH_LIST from torch; proceeding with PyTorch defaults"
    fi

    case "$SAGE_STRATEGY" in
        "mixed_with_rtx20"|"rtx20_only")
            log "Cloning Sage Attention v1.0 for RTX 20 series compatibility"
            if [ -d "SageAttention/.git" ]; then
                cd SageAttention
                git fetch --depth 1 origin || return 1
                git checkout v1.0 2>/dev/null || git checkout -b v1.0 origin/v1.0 || return 1
                git reset --hard origin/v1.0 || return 1
            else
                rm -rf SageAttention
                git clone --depth 1 https://github.com/thu-ml/SageAttention.git -b v1.0 || return 1
                cd SageAttention
            fi
            ;;
        *)
            log "Cloning latest Sage Attention for modern GPUs"
            if [ -d "SageAttention/.git" ]; then
                cd SageAttention
                git fetch --depth 1 origin || return 1
                git reset --hard origin/HEAD || return 1
            else
                rm -rf SageAttention
                git clone --depth 1 https://github.com/thu-ml/SageAttention.git || return 1
                cd SageAttention
            fi
            ;;
    esac

    log "Building Sage Attention..."
    if MAX_JOBS=$(nproc) python -m pip install --no-cache-dir --user --no-build-isolation .; then
        echo "$SAGE_STRATEGY" > "$SAGE_ATTENTION_BUILT_FLAG"
        log "Sage Attention built successfully for strategy: $SAGE_STRATEGY"
        cd "$BASE_DIR"
        return 0
    else
        log "ERROR: Sage Attention build failed"
        cd "$BASE_DIR"
        return 1
    fi
}

# Function to check if current build matches detected GPUs
needs_rebuild() {
    if [ ! -f "$SAGE_ATTENTION_BUILT_FLAG" ]; then
        return 0
    fi
    local built_strategy
    built_strategy=$(cat "$SAGE_ATTENTION_BUILT_FLAG" 2>/dev/null || echo "unknown")
    if [ "$built_strategy" != "$SAGE_STRATEGY" ]; then
        log "GPU configuration changed (was: $built_strategy, now: $SAGE_STRATEGY) - rebuild needed"
        return 0
    fi
    return 1
}

# Function to check if Sage Attention is working
test_sage_attention() {
    python -c "
import sys
try:
    import sageattention
    print('[TEST] Sage Attention import: SUCCESS')
    try:
        if hasattr(sageattention, '__version__'):
            print(f'[TEST] Version: {sageattention.__version__}')
    except:
        pass
    sys.exit(0)
except ImportError as e:
    print(f'[TEST] Sage Attention import: FAILED - {e}')
    sys.exit(1)
except Exception as e:
    print(f'[TEST] Sage Attention test: ERROR - {e}')
    sys.exit(1)
" 2>/dev/null
}

# Main GPU detection and Sage Attention setup
setup_sage_attention() {
    # Internal tracking and exported availability flag
    export SAGE_ATTENTION_BUILT=0
    export SAGE_ATTENTION_AVAILABLE=0

    # Detect GPU generations
    if ! detect_gpu_generations; then
        log "No GPUs detected, skipping Sage Attention setup"
        return 0
    fi

    # Determine optimal strategy
    determine_sage_strategy

    # Build/install if needed
    if needs_rebuild || ! test_sage_attention; then
        log "Building Sage Attention..."
        if install_triton_version && build_sage_attention_mixed && test_sage_attention; then
            export SAGE_ATTENTION_BUILT=1
            export SAGE_ATTENTION_AVAILABLE=1
            log "Sage Attention is built and importable; enable at boot by setting FORCE_SAGE_ATTENTION=1"
        else
            export SAGE_ATTENTION_BUILT=0
            export SAGE_ATTENTION_AVAILABLE=0
            log "WARNING: Sage Attention is not available after build attempt"
        fi
    else
        export SAGE_ATTENTION_BUILT=1
        export SAGE_ATTENTION_AVAILABLE=1
        log "Sage Attention already built and importable for current GPU configuration"
    fi
}

# If running as root, handle permissions and user mapping
if [ "$(id -u)" = "0" ]; then
    if [ ! -f "$PERMISSIONS_SET_FLAG" ]; then
        log "Setting up user permissions..."

        if getent group "${PGID}" >/dev/null; then
            EXISTING_GRP="$(getent group "${PGID}" | cut -d: -f1)"
            usermod -g "${EXISTING_GRP}" "${APP_USER}" || true
            APP_GROUP="${EXISTING_GRP}"
        else
            groupmod -o -g "${PGID}" "${APP_GROUP}" || true
        fi

        usermod -o -u "${PUID}" "${APP_USER}" || true

        mkdir -p "/home/${APP_USER}"
        for d in "$BASE_DIR" "/home/$APP_USER"; do
            [ -e "$d" ] && chown -R "${APP_USER}:${APP_GROUP}" "$d" || true
        done

        readarray -t PY_PATHS < <(python - <<'PY'
import sys, sysconfig, os, datetime
def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[bootstrap:python {ts}] {msg}", file=sys.stderr, flush=True)
log("Determining writable Python install targets via sysconfig.get_paths()")
keys = ("purelib","platlib","scripts","include","platinclude","data")
paths = sysconfig.get_paths()
for k in keys:
    v = paths.get(k)
    if v:
        print(v)
        log(f"emit {k} -> {v}")
d = paths.get("data")
if d:
    share = os.path.join(d, "share")
    man1 = os.path.join(share, "man", "man1")
    print(share)
    print(man1)
    log(f"emit wheel data dirs -> {share}, {man1}")
log("Finished emitting target directories")
PY
)

        for d in "${PY_PATHS[@]}"; do
            case "$d" in
                /usr/local|/usr/local/*)
                    mkdir -p "$d" || true
                    chown -R "${APP_USER}:${APP_GROUP}" "$d" || true
                    chmod -R u+rwX,g+rwX "$d" || true
                    ;;
                *) : ;;
            esac
        done

        touch "$PERMISSIONS_SET_FLAG"
        chown "${APP_USER}:${APP_GROUP}" "$PERMISSIONS_SET_FLAG"
        log "User permissions configured"
    else
        log "User permissions already configured, skipping..."
    fi

    exec runuser -u "${APP_USER}" -- "$0" "$@"
fi

# Setup Sage Attention for detected GPU configuration
setup_sage_attention

# Ensure ComfyUI-Manager exists or update it (shallow)
if [ -d "$CUSTOM_NODES_DIR/ComfyUI-Manager/.git" ]; then
    log "Updating ComfyUI-Manager in $CUSTOM_NODES_DIR/ComfyUI-Manager"
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Manager" fetch --depth 1 origin || true
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Manager" reset --hard origin/HEAD || true
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Manager" clean -fdx || true
elif [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager" ]; then
    log "Installing ComfyUI-Manager into $CUSTOM_NODES_DIR/ComfyUI-Manager"
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git "$CUSTOM_NODES_DIR/ComfyUI-Manager" || true
fi

# User-site PATHs for --user installs (custom nodes)
export PATH="$HOME/.local/bin:$PATH"
pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="$HOME/.local/lib/python${pyver}/site-packages:${PYTHONPATH:-}"

# First-run detection for custom node deps (with override)
RUN_NODE_INSTALL=0
if [ ! -f "$FIRST_RUN_FLAG" ]; then
    RUN_NODE_INSTALL=1
    log "First run detected: installing custom node dependencies"
elif [ "${COMFY_AUTO_INSTALL:-0}" = "1" ]; then
    RUN_NODE_INSTALL=1
    log "COMFY_AUTO_INSTALL=1: forcing custom node dependency install"
else
    log "Not first run and COMFY_AUTO_INSTALL!=1: skipping custom node dependency install"
fi

if [ "$RUN_NODE_INSTALL" = "1" ]; then
    log "Scanning custom nodes for requirements..."
    while IFS= read -r -d '' req; do
        log "python -m pip install --user --upgrade -r $req"
        python -m pip install --no-cache-dir --user --upgrade --upgrade-strategy only-if-needed -r "$req" || true
    done < <(find "$CUSTOM_NODES_DIR" -maxdepth 3 -type f \( -iname 'requirements.txt' -o -iname 'requirements-*.txt' -o -path '*/requirements/*.txt' \) -print0)

    while IFS= read -r -d '' pjt; do
        d="$(dirname "$pjt")"
        log "python -m pip install --user . in $d"
        (cd "$d" && python -m pip install --no-cache-dir --user .) || true
    done < <(find "$CUSTOM_NODES_DIR" -maxdepth 2 -type f -iname 'pyproject.toml' -not -path '*/ComfyUI-Manager/*' -print0)

    python -m pip check || true

    # Mark first run complete (or keep flag if already set)
    touch "$FIRST_RUN_FLAG" || true
fi

# Build ComfyUI command with Sage Attention flag only if forced
COMFYUI_ARGS=""
if [ "${FORCE_SAGE_ATTENTION:-0}" = "1" ]; then
    if test_sage_attention; then
        COMFYUI_ARGS="--use-sage-attention"
        log "Starting ComfyUI with Sage Attention forced by environment (FORCE_SAGE_ATTENTION=1)"
    else
        log "WARNING: FORCE_SAGE_ATTENTION=1 but Sage Attention import failed; starting without"
    fi
else
    if [ "${SAGE_ATTENTION_AVAILABLE:-0}" = "1" ]; then
        log "Sage Attention is built and available; set FORCE_SAGE_ATTENTION=1 to enable it on boot"
    else
        log "Sage Attention not available; starting without it"
    fi
fi

cd "$BASE_DIR"

# Handle both direct execution and passed arguments
if [ $# -eq 0 ]; then
    exec python main.py --listen 0.0.0.0 $COMFYUI_ARGS
else
    if [ "${1:-}" = "python" ] && [ "${2:-}" = "main.py" ]; then
        shift 2
        exec python main.py $COMFYUI_ARGS "$@"
    else
        exec "$@"
    fi
fi
