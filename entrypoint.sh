#!/bin/bash
set -euo pipefail

# --- config ---
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

# --- logging ---
log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Make newly created files group-writable (helps in shared volumes)
umask 0002

# --- build parallelism (single knob) ---
# Public knob: SAGE_MAX_JOBS. If unset, pick RAM/CPU heuristic.
decide_build_jobs() {
    if [ -n "${SAGE_MAX_JOBS:-}" ]; then echo "$SAGE_MAX_JOBS"; return; fi
    local mem_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    local cpu=$(nproc) cap=24 jobs
    if   [ "$mem_kb" -le $((8*1024*1024)) ];  then jobs=2
    elif [ "$mem_kb" -le $((12*1024*1024)) ]; then jobs=3
    elif [ "$mem_kb" -le $((24*1024*1024)) ]; then jobs=4
    elif [ "$mem_kb" -le $((64*1024*1024)) ]; then jobs=$(( cpu<8 ? cpu : 8 ))
    else jobs=$cpu; [ "$jobs" -gt "$cap" ] && jobs=$cap
    fi
    echo "$jobs"
}

# --- CUDA/Torch checks ---
test_pytorch_cuda() {
    python -c "
import torch, sys
if not torch.cuda.is_available():
    print('[ERROR] PyTorch CUDA not available'); sys.exit(1)
c=torch.cuda.device_count(); print(f'[TEST] PyTorch CUDA available with {c} devices')
for i in range(c):
    p=torch.cuda.get_device_properties(i)
    print(f'[TEST] GPU {i}: {p.name} (Compute {p.major}.{p.minor})')
" 2>/dev/null
}

# Determine if there is a compatible NVIDIA GPU (>= sm_75, i.e., 16-series/Turing and newer)
gpu_is_compatible() {
    python - <<'PY' 2>/dev/null
import sys
try:
    import torch
    if not torch.cuda.is_available():
        sys.exit(2)
    ok=False
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        cc=float(f"{p.major}.{p.minor}")
        if cc >= 7.5:
            ok=True
    sys.exit(0 if ok else 3)
except Exception:
    sys.exit(4)
PY
}

# Derive arch list directly from Torch; optional +PTX via SAGE_PTX_FALLBACK=1
compute_arch_list_from_torch() {
    python - <<'PY' 2>/dev/null
import os, sys
try:
    import torch
    if not torch.cuda.is_available():
        print(""); sys.exit(0)
    caps = {f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
            for i in range(torch.cuda.device_count())}
    ordered = sorted(caps, key=lambda s: tuple(int(x) for x in s.split(".")))
    if not ordered: print(""); sys.exit(0)
    if os.environ.get("SAGE_PTX_FALLBACK","0")=="1":
        highest = ordered[-1]; print(";".join(ordered+[highest + "+PTX"]))
    else:
        print(";".join(ordered))
except Exception:
    print("")
PY
}

# Fallback name-based mapping across Turing→Blackwell
detect_gpu_generations() {
    local info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "")
    local has_turing=false has_amp_ga100=false has_amp_ga10x=false has_amp_ga10b=false
    local has_ada=false has_hopper=false has_bw_cons=false has_bw_dc=false
    local n=0
    [ -z "$info" ] && { log "No NVIDIA GPUs detected"; return 1; }
    log "Detecting GPU generations:"
    while IFS= read -r g; do
        n=$((n+1)); log "  GPU $n: $g"
        case "$g" in
            *"RTX 20"*|*"T4"*) has_turing=true ;;
            *"A100"*|*"A30"*|*"A40"*) has_amp_ga100=true ;;
            *"RTX 30"*|*"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*|*"RTX 3060"*) has_amp_ga10x=true ;;
            *"Orin"*|*"Jetson"*) has_amp_ga10b=true ;;
            *"RTX 40"*|*"4090"*|*"L40"*|*"L4"*) has_ada=true ;;
            *"H100"*|*"H200"*|*"GH200"*) has_hopper=true ;;
            *"RTX 50"*|*"5090"*|*"5080"*|*"5070"*|*"5060"*|*"PRO "*Blackwell*|*"PRO 4000 Blackwell"*) has_bw_cons=true ;;
            *"B200"*|*"B100"*|*"GB200"*|*"B40"*|*"RTX 6000 Blackwell"*|*"RTX 5000 Blackwell"*) has_bw_dc=true ;;
        esac
    done <<< "$info"
    export DET_TURING=$has_turing DET_AMP80=$has_amp_ga100 DET_AMP86=$has_amp_ga10x DET_AMP87=$has_amp_ga10b
    export DET_ADA=$has_ada DET_HOPPER=$has_hopper DET_BW12=$has_bw_cons DET_BW10=$has_bw_dc
    export GPU_COUNT=$n
    log "Summary: Turing=$has_turing Amp(8.0)=$has_amp_ga100 Amp(8.6)=$has_amp_ga10x Amp(8.7)=$has_amp_ga10b Ada=$has_ada Hopper=$has_hopper Blackwell(12.x)=$has_bw_cons Blackwell(10.0)=$has_bw_dc"
    test_pytorch_cuda && log "PyTorch CUDA compatibility confirmed" || log "WARNING: PyTorch CUDA compatibility issues detected"
}

determine_sage_strategy() {
    local s=""
    if [ "${DET_TURING:-false}" = "true" ]; then
        if [ "${DET_AMP80:-false}" = "true" ] || [ "${DET_AMP86:-false}" = "true" ] || [ "${DET_AMP87:-false}" = "true" ] || [ "${DET_ADA:-false}" = "true" ] || [ "${DET_HOPPER:-false}" = "true" ] || [ "${DET_BW12:-false}" = "true" ] || [ "${DET_BW10:-false}" = "true" ]; then
            s="mixed_with_turing"; log "Mixed rig including Turing - using compatibility mode"
        else s="turing_only"; log "Turing-only rig detected"; fi
    elif [ "${DET_BW12:-false}" = "true" ] || [ "${DET_BW10:-false}" = "true" ]; then s="blackwell_capable"; log "Blackwell detected - using latest optimizations"
    elif [ "${DET_HOPPER:-false}" = "true" ]; then s="hopper_capable"; log "Hopper detected - using modern optimizations"
    elif [ "${DET_ADA:-false}" = "true" ] || [ "${DET_AMP86:-false}" = "true" ] || [ "${DET_AMP87:-false}" = "true" ] || [ "${DET_AMP80:-false}" = "true" ]; then
        s="ampere_ada_optimized"; log "Ampere/Ada detected - using standard optimizations"
    else s="fallback"; log "Unknown configuration - using fallback"; fi
    export SAGE_STRATEGY=$s
}

install_triton_version() {
    case "$SAGE_STRATEGY" in
        "mixed_with_turing"|"turing_only")
            log "Installing Triton 3.2.0 for Turing compatibility"
            python -m pip install --user --force-reinstall "triton==3.2.0" || python -m pip install --user --force-reinstall triton || true
            ;;
        "blackwell_capable"|"hopper_capable")
            log "Installing latest Triton for Hopper/Blackwell"
            python -m pip install --user --force-reinstall triton || python -m pip install --user --force-reinstall --pre triton || python -m pip install --user --force-reinstall "triton>=3.2.0" || true
            ;;
        *)
            log "Installing latest stable Triton"
            python -m pip install --user --force-reinstall triton || { log "WARNING: Triton installation failed"; return 1; }
            ;;
    esac
}

build_sage_attention_mixed() {
    log "Building Sage Attention..."
    mkdir -p "$SAGE_ATTENTION_DIR"; cd "$SAGE_ATTENTION_DIR"

    local arch_list="${SAGE_ARCH_LIST_OVERRIDE:-$(compute_arch_list_from_torch)}"
    if [ -z "$arch_list" ]; then
        local tmp=""
        [ "${DET_TURING:-false}" = "true" ] && tmp="${tmp}7.5;"
        [ "${DET_AMP80:-false}" = "true" ] && tmp="${tmp}8.0;"
        [ "${DET_AMP86:-false}" = "true" ] && tmp="${tmp}8.6;"
        [ "${DET_AMP87:-false}" = "true" ] && tmp="${tmp}8.7;"
        [ "${DET_ADA:-false}" = "true" ] && tmp="${tmp}8.9;"
        [ "${DET_HOPPER:-false}" = "true" ] && tmp="${tmp}9.0;"
        [ "${DET_BW10:-false}" = "true" ] && tmp="${tmp}10.0;"
        [ "${DET_BW12:-false}" = "true" ] && tmp="${tmp}12.0;"
        arch_list="${tmp%;}"
    fi
    export TORCH_CUDA_ARCH_LIST="$arch_list"
    log "Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

    case "$SAGE_STRATEGY" in
        "mixed_with_turing"|"turing_only")
            log "Cloning SageAttention v1.0 for Turing"
            if [ -d "SageAttention/.git" ]; then cd SageAttention; git fetch --depth 1 origin || return 1; git checkout v1.0 2>/dev/null || git checkout -b v1.0 origin/v1.0 || return 1; git reset --hard origin/v1.0 || return 1
            else rm -rf SageAttention; git clone --depth 1 https://github.com/thu-ml/SageAttention.git -b v1.0 || return 1; cd SageAttention; fi
            ;;
        *)
            log "Cloning latest SageAttention"
            if [ -d "SageAttention/.git" ]; then cd SageAttention; git fetch --depth 1 origin || return 1; git reset --hard origin/main || return 1
            else rm -rf SageAttention; git clone --depth 1 https://github.com/thu-ml/SageAttention.git || return 1; cd SageAttention; fi
            ;;
    esac

    [ "${SAGE_VERBOSE_BUILD:-0}" = "1" ] && export TORCH_CPP_BUILD_VERBOSE=1
    local jobs; jobs="$(decide_build_jobs)"
    log "Using MAX_JOBS=${jobs} for SageAttention build"

    if MAX_JOBS="${jobs}" python -m pip install --user --no-build-isolation .; then
        echo "$SAGE_STRATEGY|$TORCH_CUDA_ARCH_LIST" > "$SAGE_ATTENTION_BUILT_FLAG"
        log "SageAttention built successfully"
        cd "$BASE_DIR"; return 0
    else
        log "ERROR: SageAttention build failed"
        cd "$BASE_DIR"; return 1
    fi
}

needs_rebuild() {
    if [ ! -f "$SAGE_ATTENTION_BUILT_FLAG" ]; then return 0; fi
    local x; x=$(cat "$SAGE_ATTENTION_BUILT_FLAG" 2>/dev/null || echo "")
    local prev_strategy="${x%%|*}"; local prev_arch="${x#*|}"
    if [ "$prev_strategy" != "$SAGE_STRATEGY" ] || [ "$prev_arch" != "$TORCH_CUDA_ARCH_LIST" ]; then return 0; fi
    return 1
}

test_sage_attention() {
    python -c "
import sys
try:
    import sageattention; print('[TEST] SageAttention import: SUCCESS')
    v=getattr(sageattention,'__version__',None)
    if v: print(f'[TEST] Version: {v}'); sys.exit(0)
except ImportError as e:
    print(f'[TEST] SageAttention import: FAILED - {e}'); sys.exit(1)
except Exception as e:
    print(f'[TEST] SageAttention test: ERROR - {e}'); sys.exit(1)
" 2>/dev/null
}

setup_sage_attention() {
    export SAGE_ATTENTION_BUILT=0 SAGE_ATTENTION_AVAILABLE=0
    if ! detect_gpu_generations; then log "No GPUs detected, skipping SageAttention setup"; return 0; fi
    determine_sage_strategy

    export TORCH_CUDA_ARCH_LIST="${SAGE_ARCH_LIST_OVERRIDE:-$(compute_arch_list_from_torch)}"
    if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
        local tmp=""
        [ "${DET_TURING:-false}" = "true" ] && tmp="${tmp}7.5;"
        [ "${DET_AMP80:-false}" = "true" ] && tmp="${tmp}8.0;"
        [ "${DET_AMP86:-false}" = "true" ] && tmp="${tmp}8.6;"
        [ "${DET_AMP87:-false}" = "true" ] && tmp="${tmp}8.7;"
        [ "${DET_ADA:-false}" = "true" ] && tmp="${tmp}8.9;"
        [ "${DET_HOPPER:-false}" = "true" ] && tmp="${tmp}9.0;"
        [ "${DET_BW10:-false}" = "true" ] && tmp="${tmp}10.0;"
        [ "${DET_BW12:-false}" = "true" ] && tmp="${tmp}12.0;"
        export TORCH_CUDA_ARCH_LIST="${tmp%;}"
    fi
    log "Resolved TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

    if needs_rebuild || ! test_sage_attention; then
        log "Building SageAttention..."
        if install_triton_version && build_sage_attention_mixed && test_sage_attention; then
            export SAGE_ATTENTION_BUILT=1 SAGE_ATTENTION_AVAILABLE=1
            log "SageAttention is built; set FORCE_SAGE_ATTENTION=1 to enable it at startup"
        else
            export SAGE_ATTENTION_BUILT=0 SAGE_ATTENTION_AVAILABLE=0
            log "WARNING: SageAttention is not available after build attempt"
        fi
    else
        export SAGE_ATTENTION_BUILT=1 SAGE_ATTENTION_AVAILABLE=1
        log "SageAttention already built and importable"
    fi
}

# --- root to runtime user ---
if [ "$(id -u)" = "0" ]; then
    if [ ! -f "$PERMISSIONS_SET_FLAG" ]; then
        log "Setting up user permissions..."
        if getent group "${PGID}" >/dev/null; then
            EXISTING_GRP="$(getent group "${PGID}" | cut -d: -f1)"; usermod -g "${EXISTING_GRP}" "${APP_USER}" || true; APP_GROUP="${EXISTING_GRP}"
        else groupmod -o -g "${PGID}" "${APP_GROUP}" || true; fi
        usermod -o -u "${PUID}" "${APP_USER}" || true
        mkdir -p "/home/${APP_USER}"
        for d in "$BASE_DIR" "/home/$APP_USER"; do [ -e "$d" ] && chown -R "${APP_USER}:${APP_GROUP}" "$d" || true; done

        # Discover both system and user site dirs and make them writable by the runtime user
        readarray -t PY_PATHS < <(python - <<'PY'
import sys, sysconfig, os, site, datetime
def log(m): print(f"[bootstrap:python {datetime.datetime.now().strftime('%H:%M:%S')}] {m}", file=sys.stderr, flush=True)
log("Determining writable Python install targets via sysconfig.get_paths(), site.getsitepackages(), and site.getusersitepackages()")
seen=set()
for k in ("purelib","platlib","scripts","include","platinclude","data"):
    v = sysconfig.get_paths().get(k)
    if v and v.startswith("/usr/local") and v not in seen:
        print(v); seen.add(v); log(f"emit {k} -> {v}")
for v in (site.getusersitepackages(),):
    if v and v not in seen:
        print(v); seen.add(v); log(f"emit usersite -> {v}")
for v in site.getsitepackages():
    if v and v.startswith("/usr/local") and v not in seen:
        print(v); seen.add(v); log(f"emit sitepkg -> {v}")
d = sysconfig.get_paths().get("data")
if d:
    share=os.path.join(d,"share"); man1=os.path.join(share,"man","man1")
    for v in (share, man1):
        if v and v.startswith("/usr/local") and v not in seen:
            print(v); seen.add(v); log(f"emit wheel data -> {v}")
PY
)
        for d in "${PY_PATHS[@]}"; do
            [ -n "$d" ] || continue
            mkdir -p "$d" || true
            chown -R "${APP_USER}:${APP_GROUP}" "$d" || true
            chmod -R u+rwX,g+rwX "$d" || true
        done

        # Also ensure the main site-packages tree is writable if present (guards numpy uninstall/upgrade)
        if [ -d "/usr/local/lib/python3.12/site-packages" ]; then
            chown -R "${APP_USER}:${APP_GROUP}" /usr/local/lib/python3.12/site-packages || true
            chmod -R u+rwX,g+rwX /usr/local/lib/python3.12/site-packages || true
        fi

        touch "$PERMISSIONS_SET_FLAG"; chown "${APP_USER}:${APP_GROUP}" "$PERMISSIONS_SET_FLAG"
        log "User permissions configured"
    else
        log "User permissions already configured, skipping..."
    fi
    exec runuser -u "${APP_USER}" -- "$0" "$@"
fi

# From here on, running as $APP_USER

# Favor user installs everywhere to avoid touching system packages
export PATH="$HOME/.local/bin:$PATH"
pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="$HOME/.local/lib/python${pyver}/site-packages:${PYTHONPATH:-}"
export PIP_USER=1
export PIP_PREFER_BINARY=1

# Abort early if no compatible NVIDIA GPU (>= sm_75) is present
if ! gpu_is_compatible; then
    log "No compatible NVIDIA GPU detected (compute capability 7.5+ required). Shutting down container."
    exit 0
fi

# --- SageAttention setup (runs only if compatible GPU is present) ---
setup_sage_attention

# --- ComfyUI-Manager sync ---
if [ -d "$CUSTOM_NODES_DIR/ComfyUI-Manager/.git" ]; then
    log "Updating ComfyUI-Manager"
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Manager" fetch --depth 1 origin || true
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Manager" reset --hard origin/HEAD || true
    git -C "$CUSTOM_NODES_DIR/ComfyUI-Manager" clean -fdx || true
elif [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager" ]; then
    log "Installing ComfyUI-Manager"
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git "$CUSTOM_NODES_DIR/ComfyUI-Manager" || true
fi

# --- first-run install of custom_nodes ---
if [ ! -f "$FIRST_RUN_FLAG" ] || [ "${COMFY_FORCE_INSTALL:-0}" = "1" ]; then
    if [ "${COMFY_AUTO_INSTALL:-1}" = "1" ]; then
        log "First run or forced; installing custom node dependencies..."

        # Manager-like behavior: per-node, top-level requirements.txt only, plus optional install.py;
        shopt -s nullglob
        for d in "$CUSTOM_NODES_DIR"/*; do
            [ -d "$d" ] || continue
            base="$(basename "$d")"
            [ "$base" = "ComfyUI-Manager" ] && continue

            if [ -f "$d/requirements.txt" ]; then
                log "Installing requirements for node: $base"
                python -m pip install --no-cache-dir --user --upgrade --upgrade-strategy only-if-needed -r "$d/requirements.txt" || true
            fi

            if [ -f "$d/install.py" ]; then
                log "Running install.py for node: $base"
                (cd "$d" && python "install.py") || true
            fi
        done
        shopt -u nullglob

        python -m pip check || true
    else
        log "COMFY_AUTO_INSTALL=0; skipping dependency install"
    fi
    touch "$FIRST_RUN_FLAG"
else
    log "Not first run; skipping custom_nodes dependency install"
fi

# --- launch ComfyUI ---
COMFYUI_ARGS=""
if [ "${FORCE_SAGE_ATTENTION:-0}" = "1" ] && test_sage_attention; then
    COMFYUI_ARGS="--use-sage-attention"
    log "Starting ComfyUI with SageAttention (FORCE_SAGE_ATTENTION=1)"
else
    if [ "${SAGE_ATTENTION_AVAILABLE:-0}" = "1" ]; then
        log "SageAttention is built; set FORCE_SAGE_ATTENTION=1 to enable"
    else
        log "SageAttention not available; starting without it"
    fi
fi

cd "$BASE_DIR"
if [ $# -eq 0 ]; then
    exec python main.py --listen 0.0.0.0 $COMFYUI_ARGS
else
    if [ "$1" = "python" ] && [ "${2:-}" = "main.py" ]; then
        shift 2; exec python main.py $COMFYUI_ARGS "$@"
    else
        exec "$@"
    fi
fi
