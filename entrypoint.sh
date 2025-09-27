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
FIRST_RUN_FLAG="$BASE_DIR/.first_run_done"

# Function to log with timestamp
log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Function to test PyTorch CUDA compatibility
test_pytorch_cuda() {
    python -c "
import torch, sys
if not torch.cuda.is_available():
    print('[ERROR] PyTorch CUDA not available')
    sys.exit(1)
c = torch.cuda.device_count()
print(f'[TEST] PyTorch CUDA available with {c} devices')
for i in range(c):
    props = torch.cuda.get_device_properties(i)
    print(f'[TEST] GPU {i}: {props.name} (Compute {props.major}.{props.minor})')
" 2>/dev/null
}

# Function to detect all GPUs and their generations
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
    if [ "$DETECTED_RTX20" = "true" ]; then
        if [ "$DETECTED_RTX30" = "true" ] || [ "$DETECTED_RTX40" = "true" ] || [ "$DETECTED_RTX50" = "true" ]; then
            strategy="mixed_with_rtx20"
            log "Mixed GPU setup detected with RTX 20 series - using compatibility mode"
        else
            strategy="rtx20_only"
            log "RTX 20 series only detected"
        fi
    elif [ "$DETECTED_RTX50" = "true" ]; then
        strategy="rtx50_capable"
        log "RTX 50 series detected - using latest optimizations"
    elif [ "$DETECTED_RTX40" = "true" ] || [ "$DETECTED_RTX30" = "true" ]; then
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
            log "Installing Triton 3.2.0 for broader compatibility on Turing-era GPUs"
            python -m pip install --user --force-reinstall "triton==3.2.0" || {
                log "WARNING: Failed to pin Triton 3.2.0, trying latest"
                python -m pip install --user --force-reinstall triton || true
            }
            ;;
        "rtx50_capable")
            log "Installing latest Triton for Blackwell/RTX 50"
            python -m pip install --user --force-reinstall triton || \
            python -m pip install --user --force-reinstall --pre triton || {
                log "WARNING: Latest Triton install failed, falling back to >=3.2.0"
                python -m pip install --user --force-reinstall "triton>=3.2.0" || true
            }
            ;;
        *)
            log "Installing latest stable Triton"
            python -m pip install --user --force-reinstall triton || {
                log "WARNING: Triton installation failed, continuing without"
                return 1
            }
            ;;
    esac
}

# Function to build Sage Attention with architecture-specific optimizations
build_sage_attention_mixed() {
    log "Building Sage Attention for current GPU environment..."
    mkdir -p "$SAGE_ATTENTION_DIR"
    cd "$SAGE_ATTENTION_DIR"

    # Compute capability mapping for TORCH_CUDA_ARCH_LIST:
    # Turing = 7.5, Ampere = 8.6, Ada = 8.9, Blackwell (RTX 50) = 10.0
    # See NVIDIA Blackwell guide (sm_100/compute_100 ~ 10.0) and PyTorch arch list semantics. [doc refs in text]
    local cuda_arch_list=""
    [ "$DETECTED_RTX20" = "true" ] && cuda_arch_list="${cuda_arch_list}7.5;"
    [ "$DETECTED_RTX30" = "true" ] && cuda_arch_list="${cuda_arch_list}8.6;"
    [ "$DETECTED_RTX40" = "true" ] && cuda_arch_list="${cuda_arch_list}8.9;"
    [ "$DETECTED_RTX50" = "true" ] && cuda_arch_list="${cuda_arch_list}10.0;"
    cuda_arch_list=${cuda_arch_list%;}

    export TORCH_CUDA_ARCH_LIST="$cuda_arch_list"
    log "Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

    case "$SAGE_STRATEGY" in
        "mixed_with_rtx20"|"rtx20_only")
            log "Cloning SageAttention v1.0 for RTX 20 series compatibility"
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
            log "Cloning latest SageAttention for modern GPUs"
            if [ -d "SageAttention/.git" ]; then
                cd SageAttention
                git fetch --depth 1 origin || return 1
                git reset --hard origin/main || return 1
            else
                rm -rf SageAttention
                git clone --depth 1 https://github.com/thu-ml/SageAttention.git || return 1
                cd SageAttention
            fi
            ;;
    esac

    log "Building SageAttention (no-build-isolation) ..."
    if MAX_JOBS=$(nproc) python -m pip install --user --no-build-isolation .; then
        echo "$SAGE_STRATEGY" > "$SAGE_ATTENTION_BUILT_FLAG"
        log "SageAttention built successfully for strategy: $SAGE_STRATEGY"
        cd "$BASE_DIR"
        return 0
    else
        log "ERROR: SageAttention build failed"
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

# Function to check if SageAttention is working
test_sage_attention() {
    python -c "
import sys
try:
    import sageattention
    print('[TEST] SageAttention import: SUCCESS')
    try:
        v = getattr(sageattention, '__version__', None)
        if v: print(f'[TEST] Version: {v}')
    except:
        pass
    sys.exit(0)
except ImportError as e:
    print(f'[TEST] SageAttention import: FAILED - {e}')
    sys.exit(1)
except Exception as e:
    print(f'[TEST] SageAttention test: ERROR - {e}')
    sys.exit(1)
" 2>/dev/null
}

# Main GPU detection and SageAttention setup
setup_sage_attention() {
    # Export build-visible status flags
    export SAGE_ATTENTION_BUILT=0
    export SAGE_ATTENTION_AVAILABLE=0

    if ! detect_gpu_generations; then
        log "No GPUs detected, skipping SageAttention setup"
        return 0
    fi

    determine_sage_strategy

    if needs_rebuild || ! test_sage_attention; then
        log "Building SageAttention..."
        if install_triton_version && build_sage_attention_mixed && test_sage_attention; then
            export SAGE_ATTENTION_BUILT=1
            export SAGE_ATTENTION_AVAILABLE=1
            log "SageAttention is built; set FORCE_SAGE_ATTENTION=1 to enable it at startup"
        else
            export SAGE_ATTENTION_BUILT=0
            export SAGE_ATTENTION_AVAILABLE=0
            log "WARNING: SageAttention is not available after build attempt"
        fi
    else
        export SAGE_ATTENTION_BUILT=1
        export SAGE_ATTENTION_AVAILABLE=1
        log "SageAttention already built and importable for current GPU configuration"
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
    import datetime
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

# Setup SageAttention for detected GPU configuration
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

# First-run driven auto-install of custom node deps
if [ ! -f "$FIRST_RUN_FLAG" ] || [ "${COMFY_FORCE_INSTALL:-0}" = "1" ]; then
    if [ "${COMFY_AUTO_INSTALL:-1}" = "1" ]; then
        log "First run detected or forced; scanning custom nodes for requirements..."
        # requirements*.txt
        while IFS= read -r -d '' req; do
            log "python -m pip install --user --upgrade -r $req"
            python -m pip install --no-cache-dir --user --upgrade --upgrade-strategy only-if-needed -r "$req" || true
        done < <(find "$CUSTOM_NODES_DIR" -maxdepth 3 -type f \( -iname 'requirements.txt' -o -iname 'requirements-*.txt' -o -path '*/requirements/*.txt' \) -print0)

        # pyproject.toml (exclude ComfyUI-Manager)
        while IFS= read -r -d '' pjt; do
            d="$(dirname "$pjt")"
            log "python -m pip install --user . in $d"
            (cd "$d" && python -m pip install --no-cache-dir --user .) || true
        done < <(find "$CUSTOM_NODES_DIR" -maxdepth 2 -type f -iname 'pyproject.toml' -not -path '*/ComfyUI-Manager/*' -print0)

        python -m pip check || true
    else
        log "COMFY_AUTO_INSTALL=0; skipping dependency install on first run"
    fi
    touch "$FIRST_RUN_FLAG"
else
    log "Not first run; skipping custom_nodes dependency install"
fi

# Build ComfyUI command with SageAttention usage controlled only by FORCE_SAGE_ATTENTION
COMFYUI_ARGS=""
if [ "${FORCE_SAGE_ATTENTION:-0}" = "1" ]; then
    if test_sage_attention; then
        COMFYUI_ARGS="--use-sage-attention"
        log "Starting ComfyUI with SageAttention enabled by environment (FORCE_SAGE_ATTENTION=1)"
    else
        log "WARNING: FORCE_SAGE_ATTENTION=1 but SageAttention import failed; starting without"
    fi
else
    if [ "${SAGE_ATTENTION_AVAILABLE:-0}" = "1" ]; then
        log "SageAttention is built; set FORCE_SAGE_ATTENTION=1 to enable it at startup"
    else
        log "SageAttention not available; starting without it"
    fi
fi

cd "$BASE_DIR"

# Handle both direct execution and passed arguments
if [ $# -eq 0 ]; then
    exec python main.py --listen 0.0.0.0 $COMFYUI_ARGS
else
    if [ "$1" = "python" ] && [ "${2:-}" = "main.py" ]; then
        shift 2
        exec python main.py $COMFYUI_ARGS "$@"
    else
        exec "$@"
    fi
fi
