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

# --- unified GPU probe (torch-based) ---
probe_and_prepare_gpu() {
python - <<'PY' 2>/dev/null
import os, sys
try:
    import torch
except Exception:
    print("GPU_COUNT=0"); print("COMPAT_GE_75=0"); print("TORCH_CUDA_ARCH_LIST=''")
    print("DET_TURING=false"); print("DET_AMP80=false"); print("DET_AMP86=false"); print("DET_AMP87=false")
    print("DET_ADA=false"); print("DET_HOPPER=false"); print("DET_BW12=false"); print("DET_BW10=false")
    print("SAGE_STRATEGY='fallback'"); sys.exit(0)
if not torch.cuda.is_available():
    print("GPU_COUNT=0"); print("COMPAT_GE_75=0"); print("TORCH_CUDA_ARCH_LIST=''")
    print("DET_TURING=false"); print("DET_AMP80=false"); print("DET_AMP86=false"); print("DET_AMP87=false")
    print("DET_ADA=false"); print("DET_HOPPER=false"); print("DET_BW12=false"); print("DET_BW10=false")
    print("SAGE_STRATEGY='fallback'"); sys.exit(0)
n = torch.cuda.device_count()
ccs = []
flags = {"DET_TURING":False,"DET_AMP80":False,"DET_AMP86":False,"DET_AMP87":False,"DET_ADA":False,"DET_HOPPER":False,"DET_BW12":False,"DET_BW10":False}
compat = False
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    mj, mn = p.major, p.minor
    ccs.append(f"{mj}.{mn}")
    if (mj,mn)==(7,5): flags["DET_TURING"]=True
    elif (mj,mn)==(8,0): flags["DET_AMP80"]=True
    elif (mj,mn)==(8,6): flags["DET_AMP86"]=True
    elif (mj,mn)==(8,7): flags["DET_AMP87"]=True
    elif (mj,mn)==(8,9): flags["DET_ADA"]=True
    elif (mj,mn)==(9,0): flags["DET_HOPPER"]=True
    elif (mj,mn)==(10,0): flags["DET_BW10"]=True
    elif (mj,mn)==(12,0): flags["DET_BW12"]=True
    if (mj*10+mn) >= 75:
        compat = True
ordered = sorted(set(ccs), key=lambda s: tuple(map(int, s.split("."))))
arch_list = ";".join(ordered) if ordered else ""
if flags["DET_TURING"]:
    if any(flags[k] for k in ["DET_AMP80","DET_AMP86","DET_AMP87","DET_ADA","DET_HOPPER","DET_BW12","DET_BW10"]):
        strategy = "mixed_with_turing"
    else:
        strategy = "turing_only"
elif flags["DET_BW12"] or flags["DET_BW10"]:
    strategy = "blackwell_capable"
elif flags["DET_HOPPER"]:
    strategy = "hopper_capable"
elif flags["DET_ADA"] or flags["DET_AMP86"] or flags["DET_AMP87"] or flags["DET_AMP80"]:
    strategy = "ampere_ada_optimized"
else:
    strategy = "fallback"
print(f"GPU_COUNT={n}")
print(f"COMPAT_GE_75={1 if compat else 0}")
print(f"TORCH_CUDA_ARCH_LIST='{arch_list}'")
for k,v in flags.items():
    print(f"{k}={'true' if v else 'false'}")
print(f"SAGE_STRATEGY='{strategy}'")
print(f"[GPU] Found {n} CUDA device(s); CC list: {arch_list or 'none'}; strategy={strategy}; compat>={7.5}:{compat}", file=sys.stderr)
PY
}

# --- install triton versions based on strategy ---
install_triton_version() {
    case "${SAGE_STRATEGY:-fallback}" in
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
    export TORCH_CUDA_ARCH_LIST="${SAGE_ARCH_LIST_OVERRIDE:-${TORCH_CUDA_ARCH_LIST:-}}"
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
    fi
    log "Set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

    case "${SAGE_STRATEGY:-fallback}" in
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
        echo "${SAGE_STRATEGY:-fallback}|${TORCH_CUDA_ARCH_LIST:-}" > "$SAGE_ATTENTION_BUILT_FLAG"
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
    if [ "$prev_strategy" != "${SAGE_STRATEGY:-fallback}" ] || [ "$prev_arch" != "${TORCH_CUDA_ARCH_LIST:-}" ]; then return 0; fi
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
    if [ "${GPU_COUNT:-0}" -eq 0 ]; then log "No GPUs detected, skipping SageAttention setup"; return 0; fi
    if [ "${COMPAT_GE_75:-0}" -ne 1 ]; then log "GPU compute capability < 7.5; skipping SageAttention"; return 0; fi
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

export PATH="$HOME/.local/bin:$PATH"
pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="$HOME/.local/lib/python${pyver}/site-packages:${PYTHONPATH:-}"
export PIP_USER=1
export PIP_PREFER_BINARY=1

# --- single GPU probe + early exit ---
eval "$(probe_and_prepare_gpu)"
if [ "${GPU_COUNT:-0}" -eq 0 ] || [ "${COMPAT_GE_75:-0}" -ne 1 ]; then
    log "No compatible NVIDIA GPU (compute capability >= 7.5) detected; shutting down."
    exit 0
fi

# --- Ensure package manager and Manager deps are available ---
# Ensure python -m pip works (bootstrap if needed)
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip --version >/dev/null 2>&1 || log "WARNING: pip still not available after ensurepip"

# Ensure ComfyUI-Manager minimal Python deps
python - <<'PY' || python -m pip install --no-cache-dir --user toml || true
import sys
try:
    import toml  # noqa
    sys.exit(0)
except Exception:
    sys.exit(1)
PY

# --- SageAttention setup using probed data ---
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

# --- Ensure ONNX Runtime has CUDA provider (GPU) ---
python - <<'PY' || {
import sys
try:
    import onnxruntime as ort
    ok = "CUDAExecutionProvider" in ort.get_available_providers()
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
    log "Installing onnxruntime-gpu for CUDAExecutionProvider..."
    python -m pip uninstall -y onnxruntime || true
    python -m pip install --no-cache-dir --user "onnxruntime-gpu>=1.19" || true
    python - <<'P2' || log "WARNING: ONNX Runtime CUDA provider not available after installation"
import onnxruntime as ort, sys
print("ORT providers:", ort.get_available_providers())
sys.exit(0 if "CUDAExecutionProvider" in ort.get_available_providers() else 1)
P2

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
