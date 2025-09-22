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

# Function to log with timestamp
log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# Function to detect all GPUs and their generations
detect_gpu_generations() {
    local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "")
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
            *"RTX 20"*|*"2060"*|*"2070"*|*"2080"*|*"2090"*) 
                has_rtx20=true
                ;;
            *"RTX 30"*|*"3060"*|*"3070"*|*"3080"*|*"3090"*) 
                has_rtx30=true
                ;;
            *"RTX 40"*|*"4060"*|*"4070"*|*"4080"*|*"4090"*) 
                has_rtx40=true
                ;;
            *"RTX 50"*|*"5060"*|*"5070"*|*"5080"*|*"5090"*) 
                has_rtx50=true
                ;;
        esac
    done <<< "$gpu_info"
    
    # Store detection results globally
    export DETECTED_RTX20=$has_rtx20
    export DETECTED_RTX30=$has_rtx30  
    export DETECTED_RTX40=$has_rtx40
    export DETECTED_RTX50=$has_rtx50
    export GPU_COUNT=$gpu_count
    
    log "Detection summary: RTX20=$has_rtx20, RTX30=$has_rtx30, RTX40=$has_rtx40, RTX50=$has_rtx50"
}

# Function to determine optimal Sage Attention strategy for mixed GPUs
determine_sage_strategy() {
    local strategy=""
    
    # Mixed generation logic - prioritize compatibility over peak performance
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
            log "Installing Triton 3.2.0 for RTX 20 series compatibility"
            python -m pip install --force-reinstall triton==3.2.0
            ;;
        "rtx50_capable")
            log "Installing latest Triton for RTX 50 series"
            python -m pip install --force-reinstall triton
            ;;
        "rtx30_40_optimized")
            log "Installing optimal Triton for RTX 30/40 series"
            python -m pip install --force-reinstall triton
            ;;
        *)
            log "Installing default Triton version"
            python -m pip install --force-reinstall triton
            ;;
    esac
}

# Function to build Sage Attention with architecture-specific optimizations
build_sage_attention_mixed() {
    log "Building Sage Attention for mixed GPU environment..."
    
    # Create sage attention directory
    mkdir -p "$SAGE_ATTENTION_DIR"
    cd "$SAGE_ATTENTION_DIR"
    
    # Set CUDA architecture list based on detected GPUs
    local cuda_arch_list=""
    [ "$DETECTED_RTX20" = "true" ] && cuda_arch_list="${cuda_arch_list}7.5;"
    [ "$DETECTED_RTX30" = "true" ] && cuda_arch_list="${cuda_arch_list}8.6;"  
    [ "$DETECTED_RTX40" = "true" ] && cuda_arch_list="${cuda_arch_list}8.9;"
    [ "$DETECTED_RTX50" = "true" ] && cuda_arch_list="${cuda_arch_list}9.0;"
    
    # Remove trailing semicolon
    cuda_arch_list=${cuda_arch_list%;}
    
    export TORCH_CUDA_ARCH_LIST="$cuda_arch_list"
    log "Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    
    # Clone or update repository based on strategy
    case "$SAGE_STRATEGY" in
        "mixed_with_rtx20"|"rtx20_only")
            log "Cloning Sage Attention v1.0 for RTX 20 series compatibility"
            if [ -d "SageAttention/.git" ]; then
                cd SageAttention
                git fetch --depth 1 origin
                git checkout v1.0 2>/dev/null || git checkout -b v1.0 origin/v1.0
                git reset --hard origin/v1.0
            else
                rm -rf SageAttention
                git clone --depth 1 https://github.com/thu-ml/SageAttention.git -b v1.0
                cd SageAttention
            fi
            ;;
        *)
            log "Cloning latest Sage Attention for modern GPUs"
            if [ -d "SageAttention/.git" ]; then
                cd SageAttention  
                git fetch --depth 1 origin
                git reset --hard origin/main
            else
                rm -rf SageAttention
                git clone --depth 1 https://github.com/thu-ml/SageAttention.git
                cd SageAttention
            fi
            ;;
    esac
    
    # Build with architecture-specific flags
    log "Building Sage Attention with multi-GPU support..."
    if MAX_JOBS=$(nproc) python setup.py install; then
        # Create strategy-specific built flag
        echo "$SAGE_STRATEGY" > "$SAGE_ATTENTION_BUILT_FLAG"
        log "Sage Attention built successfully for strategy: $SAGE_STRATEGY"
        return 0
    else
        log "ERROR: Sage Attention build failed"
        return 1
    fi
    
    cd "$BASE_DIR"
}

# Function to check if current build matches detected GPUs
needs_rebuild() {
    if [ ! -f "$SAGE_ATTENTION_BUILT_FLAG" ]; then
        return 0  # Needs build
    fi
    
    local built_strategy=$(cat "$SAGE_ATTENTION_BUILT_FLAG" 2>/dev/null || echo "unknown")
    if [ "$built_strategy" != "$SAGE_STRATEGY" ]; then
        log "GPU configuration changed (was: $built_strategy, now: $SAGE_STRATEGY) - rebuild needed"
        return 0  # Needs rebuild
    fi
    
    return 1  # No rebuild needed
}

# Function to check if Sage Attention is working
test_sage_attention() {
    python -c "
import sys
try:
    import sageattention
    print('[TEST] Sage Attention import: SUCCESS')
    
    # Try to get version info
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
    # Initialize Sage Attention availability flag
    export SAGE_ATTENTION_AVAILABLE=0
    
    # Detect GPU generations
    if ! detect_gpu_generations; then
        log "No GPUs detected, skipping Sage Attention setup"
        return 0
    fi
    
    # Determine optimal strategy
    determine_sage_strategy
    
    # Check if rebuild is needed
    if needs_rebuild || ! test_sage_attention; then
        log "Building Sage Attention..."
        
        # Install appropriate Triton version first
        install_triton_version
        
        # Build Sage Attention
        if build_sage_attention_mixed; then
            # Test installation
            if test_sage_attention; then
                export SAGE_ATTENTION_AVAILABLE=1
                log "Sage Attention setup completed successfully"
                log "SAGE_ATTENTION_AVAILABLE=1 (will use --use-sage-attention flag)"
            else
                log "WARNING: Sage Attention build succeeded but import test failed"
                export SAGE_ATTENTION_AVAILABLE=0
            fi
        else
            log "ERROR: Sage Attention build failed"
            export SAGE_ATTENTION_AVAILABLE=0
        fi
    else
        export SAGE_ATTENTION_AVAILABLE=1
        log "Sage Attention already built and working for current GPU configuration"
        log "SAGE_ATTENTION_AVAILABLE=1 (will use --use-sage-attention flag)"
    fi
}

# If running as root, handle permissions and user mapping
if [ "$(id -u)" = "0" ]; then
    # Check if permissions are already set
    if [ ! -f "$PERMISSIONS_SET_FLAG" ]; then
        log "Setting up user permissions..."
        
        # Map group to PGID if it already exists, otherwise remap the named group
        if getent group "${PGID}" >/dev/null; then
            EXISTING_GRP="$(getent group "${PGID}" | cut -d: -f1)"
            usermod -g "${EXISTING_GRP}" "${APP_USER}" || true
            APP_GROUP="${EXISTING_GRP}"
        else
            groupmod -o -g "${PGID}" "${APP_GROUP}" || true
        fi
        
        # Map user to PUID
        usermod -o -u "${PUID}" "${APP_USER}" || true
        
        # Ensure home and app dir exist and are owned
        mkdir -p "/home/${APP_USER}"
        for d in "$BASE_DIR" "/home/$APP_USER"; do
            [ -e "$d" ] && chown -R "${APP_USER}:${APP_GROUP}" "$d" || true
        done
        
        # Make Python system install targets writable for the runtime user (only under /usr/local)
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
        
        # Make directories writable
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
        
        # Create permissions set flag
        touch "$PERMISSIONS_SET_FLAG"
        chown "${APP_USER}:${APP_GROUP}" "$PERMISSIONS_SET_FLAG"
        log "User permissions configured"
    else
        log "User permissions already configured, skipping..."
    fi
    
    # Re-exec as the runtime user
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

# Auto-install custom node deps
if [ "${COMFY_AUTO_INSTALL:-1}" = "1" ]; then
    log "Scanning custom nodes for requirements..."
    # Install any requirements*.txt found under custom_nodes (upgrade within constraints)
    while IFS= read -r -d '' req; do
        log "pip install --user --upgrade -r $req"
        pip install --no-cache-dir --user --upgrade --upgrade-strategy only-if-needed -r "$req" || true
    done < <(find "$CUSTOM_NODES_DIR" -maxdepth 3 -type f \( -iname 'requirements.txt' -o -iname 'requirements-*.txt' -o -path '*/requirements/*.txt' \) -print0)
    
    # For pyproject.toml-based nodes, EXCLUDE ComfyUI-Manager (it's not meant to be wheel-built)
    while IFS= read -r -d '' pjt; do
        d="$(dirname "$pjt")"
        log "pip install --user . in $d"
        (cd "$d" && pip install --no-cache-dir --user .) || true
    done < <(find "$CUSTOM_NODES_DIR" -maxdepth 2 -type f -iname 'pyproject.toml' -not -path '*/ComfyUI-Manager/*' -print0)
    
    pip check || true
fi

# Build ComfyUI command with Sage Attention flag if available
COMFYUI_ARGS=""
if [ "${SAGE_ATTENTION_AVAILABLE:-0}" = "1" ]; then
    COMFYUI_ARGS="--use-sage-attention"
    log "Starting ComfyUI with Sage Attention enabled"
else
    log "Starting ComfyUI without Sage Attention (not available or build failed)"
fi

cd "$BASE_DIR"

# Handle both direct execution and passed arguments
if [ $# -eq 0 ]; then
    # No arguments passed, use default
    exec python main.py --listen 0.0.0.0 $COMFYUI_ARGS
else
    # Arguments were passed, check if it's the default command
    if [ "$1" = "python" ] && [ "$2" = "main.py" ]; then
        # Default python command, add our args
        shift 2  # Remove 'python main.py'
        exec python main.py $COMFYUI_ARGS "$@"
    else
        # Custom command, pass through as-is
        exec "$@"
    fi
fi
