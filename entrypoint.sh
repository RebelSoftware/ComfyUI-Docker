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

# Manager config (persistent) - use new location, not legacy default/ComfyUI-Manager
USER_DIR="$BASE_DIR/user"
CM_CFG_DIR="$USER_DIR/__manager"
CM_CFG="$CM_CFG_DIR/config.ini"
CM_SEEDED_FLAG="$CM_CFG_DIR/.config_seeded"

# --- logging ---
log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Make newly created files group-writable (helps in shared volumes)
umask 0002

# --- build parallelism ---
decide_build_jobs() {
    if [ -n "${SAGE_MAX_JOBS:-}" ]; then echo "$SAGE_MAX_JOBS"; return; fi
    local mem_kb; mem_kb=$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    local cpu; cpu=$(nproc); local cap=24; local jobs
    if   [ "$mem_kb" -le $((8*1024*1024)) ];  then jobs=2
    elif [ "$mem_kb" -le $((12*1024*1024)) ]; then jobs=3
    elif [ "$mem_kb" -le $((24*1024*1024)) ]; then jobs=4
    elif [ "$mem_kb" -le $((64*1024*1024)) ]; then jobs=$(( cpu<8 ? cpu : 8 ))
    else jobs=$cpu; [ "$jobs" -gt "$cap" ] && jobs=$cap
    fi
    echo "$jobs"
}

# --- GPU probe (torch-based) ---
probe_gpu() {
python - <<'PY' 2>/dev/null
import sys
try:
    import torch
except Exception:
    print("GPU_COUNT=0"); print("COMPAT_GE_75=0"); print("TORCH_CUDA_ARCH_LIST="); print("SAGE_STRATEGY=fallback"); sys.exit(0)
if not torch.cuda.is_available():
    print("GPU_COUNT=0"); print("COMPAT_GE_75=0"); print("TORCH_CUDA_ARCH_LIST="); print("SAGE_STRATEGY=fallback"); sys.exit(0)
n = torch.cuda.device_count()
ccs = []; compat = False; has_turing = False; has_ampere_plus = False
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    mj, mn = p.major, p.minor
    ccs.append(f"{mj}.{mn}")
    if (mj*10+mn) >= 75: compat = True
    if (mj, mn) == (7, 5): has_turing = True
    if mj >= 8: has_ampere_plus = True
ordered = sorted(set(ccs), key=lambda s: tuple(map(int, s.split("."))))
arch_list = ";".join(ordered)
if has_turing and has_ampere_plus: strategy = "mixed_with_turing"
elif has_turing: strategy = "turing_only"
else: strategy = "ampere_ada_or_newer"
print(f"GPU_COUNT={n}")
print(f"COMPAT_GE_75={1 if compat else 0}")
print(f"TORCH_CUDA_ARCH_LIST={arch_list}")
print(f"SAGE_STRATEGY={strategy}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"[GPU] cuda:{i} - {p.name} (CC {p.major}.{p.minor})", file=sys.stderr)
PY
}

# --- SageAttention ---
needs_sage_rebuild() {
    [ ! -f "$SAGE_ATTENTION_BUILT_FLAG" ] && return 0
    local stored; stored=$(cat "$SAGE_ATTENTION_BUILT_FLAG" 2>/dev/null || echo "")
    local prev_strategy="${stored%%|*}"; local prev_arch="${stored#*|}"
    [ "$prev_strategy" != "${SAGE_STRATEGY:-fallback}" ] && return 0
    [ "$prev_arch" != "${TORCH_CUDA_ARCH_LIST:-}" ] && return 0
    return 1
}

test_sage_attention() {
    python -c "import sageattention; print('[TEST] SageAttention import: OK')" 2>/dev/null
}

build_sage_attention() {
    log "Building SageAttention (strategy=${SAGE_STRATEGY:-fallback})..."
    mkdir -p "$SAGE_ATTENTION_DIR"; cd "$SAGE_ATTENTION_DIR"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0;12.0}"

    # Turing (SM 7.5) requires the v1.0 branch; newer GPUs use main
    case "${SAGE_STRATEGY:-fallback}" in
        "mixed_with_turing"|"turing_only")
            log "Cloning SageAttention v1.0 (Turing compatibility)"
            if [ -d "SageAttention/.git" ]; then
                cd SageAttention; git fetch --depth 1 origin || return 1
                git checkout v1.0 2>/dev/null || git checkout -b v1.0 origin/v1.0 || return 1
                git reset --hard origin/v1.0 || return 1
            else
                rm -rf SageAttention
                git clone --depth 1 https://github.com/thu-ml/SageAttention.git -b v1.0 || return 1
                cd SageAttention
            fi
            # Turing needs Triton 3.2.0
            local cur_triton; cur_triton=$(python -c "import importlib.metadata as m; print(m.version('triton'))" 2>/dev/null || echo "")
            if [ "$cur_triton" != "3.2.0" ]; then
                log "Installing Triton 3.2.0 for Turing (current: ${cur_triton:-none})"
                python -m pip install --no-cache-dir "triton==3.2.0" || true
            fi
            ;;
        *)
            log "Cloning SageAttention (latest)"
            if [ -d "SageAttention/.git" ]; then
                cd SageAttention; git fetch --depth 1 origin || return 1
                git reset --hard origin/main || return 1
            else
                rm -rf SageAttention
                git clone --depth 1 https://github.com/thu-ml/SageAttention.git || return 1
                cd SageAttention
            fi
            ;;
    esac

    local jobs; jobs="$(decide_build_jobs)"
    log "Compiling with MAX_JOBS=${jobs}"
    if MAX_JOBS="${jobs}" python -m pip install --no-build-isolation .; then
        echo "${SAGE_STRATEGY:-fallback}|${TORCH_CUDA_ARCH_LIST:-}" > "$SAGE_ATTENTION_BUILT_FLAG"
        cd "$SAGE_ATTENTION_DIR"; rm -rf SageAttention || true
        cd "$BASE_DIR"
        log "SageAttention built successfully"
        return 0
    else
        cd "$BASE_DIR"
        log "WARNING: SageAttention build failed"
        return 1
    fi
}

setup_sage_attention() {
    export SAGE_ATTENTION_AVAILABLE=0
    if [ "${GPU_COUNT:-0}" -eq 0 ] || [ "${COMPAT_GE_75:-0}" -ne 1 ]; then
        log "SageAttention: skipped (no compatible GPU)"
        return 0
    fi
    if needs_sage_rebuild || ! test_sage_attention 2>/dev/null; then
        if build_sage_attention && test_sage_attention 2>/dev/null; then
            export SAGE_ATTENTION_AVAILABLE=1
            log "SageAttention ready; set FORCE_SAGE_ATTENTION=1 to enable"
        fi
    else
        export SAGE_ATTENTION_AVAILABLE=1
        log "SageAttention already built and importable"
    fi
}

# --- ComfyUI-Manager config from CM_* env ---
configure_manager_config() {
    # Skip if already properly seeded and no CM_* env vars changed
    if [ -f "$CM_SEEDED_FLAG" ] && [ -f "$CM_CFG" ]; then
        return 0
    fi
    
python - "$CM_CFG" "$CM_SEEDED_FLAG" <<'PY'
import os, sys, configparser, pathlib
cfg_path = pathlib.Path(sys.argv[1])
seed_flag = pathlib.Path(sys.argv[2])
cfg_dir = cfg_path.parent
cfg_dir.mkdir(parents=True, exist_ok=True)

def norm_bool(v:str):
    t=v.strip().lower()
    if t in ("1","true","yes","on"): return "True"
    if t in ("0","false","no","off"): return "False"
    return v

env_items = {}
for k,v in os.environ.items():
    if not k.startswith("CM_"): continue
    key = k[3:].lower()
    env_items[key] = norm_bool(v)

cfg = configparser.ConfigParser()
first_seed = not seed_flag.exists()
if cfg_path.exists():
    cfg.read(cfg_path)
if "default" not in cfg:
    cfg["default"] = {}
if first_seed:
    cfg["default"].clear()
    for k,v in sorted(env_items.items()):
        cfg["default"][k] = v
    cfg_path.write_text("", encoding="utf-8")
    with cfg_path.open("w", encoding="utf-8") as f:
        cfg.write(f)
    seed_flag.touch()
    print(f"[CFG] created: {cfg_path} with {len(env_items)} CM_ keys", file=sys.stderr)
else:
    for k,v in env_items.items():
        if cfg["default"].get(k) != v:
            cfg["default"][k] = v
    tmp = cfg_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        cfg.write(f)
    tmp.replace(cfg_path)
    print(f"[CFG] updated: {cfg_path} applied {len(env_items)} CM_ keys", file=sys.stderr)
PY
}

# --- root: set up permissions then drop to appuser ---
if [ "$(id -u)" = "0" ]; then
    # GPU probe (needed for SageAttention strategy)
    eval "$(probe_gpu)"
    export GPU_COUNT COMPAT_GE_75 TORCH_CUDA_ARCH_LIST SAGE_STRATEGY
    log "GPU probe: ${GPU_COUNT:-0} device(s); arch=${TORCH_CUDA_ARCH_LIST:-none}; strategy=${SAGE_STRATEGY:-fallback}"

    if [ ! -f "$PERMISSIONS_SET_FLAG" ]; then
        log "Setting up user permissions..."
        if getent group "${PGID}" >/dev/null; then
            EXISTING_GRP="$(getent group "${PGID}" | cut -d: -f1)"; usermod -g "${EXISTING_GRP}" "${APP_USER}" || true; APP_GROUP="${EXISTING_GRP}"
        else groupmod -o -g "${PGID}" "${APP_GROUP}" || true; fi
        usermod -o -u "${PUID}" "${APP_USER}" || true
        mkdir -p "/home/${APP_USER}"
        for d in "$BASE_DIR" "/home/$APP_USER"; do [ -e "$d" ] && chown -R "${APP_USER}:${APP_GROUP}" "$d" || true; done

        readarray -t PY_PATHS < <(python - <<'PY'
import sys, sysconfig, os, site
seen=set()
for k in ("purelib","platlib","scripts","include","platinclude","data"):
    v = sysconfig.get_paths().get(k)
    if v and v.startswith("/usr/local") and v not in seen:
        print(v); seen.add(v)
for v in (site.getusersitepackages(),):
    if v and v not in seen:
        print(v); seen.add(v)
for v in site.getsitepackages():
    if v and v.startswith("/usr/local") and v not in seen:
        print(v); seen.add(v)
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

    exec runuser -p -u "${APP_USER}" -- "$0" "$@"
fi

# --- From here on, running as $APP_USER ---

# --- Git configuration (prevent permission warnings) ---
export GIT_CONFIG_GLOBAL="/home/${APP_USER}/.gitconfig"
git config --global core.safecround.mode=false 2>/dev/null || true

# --- SageAttention setup ---
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

# Ensure manager config directory exists
mkdir -p "$CM_CFG_DIR"

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
                python -m pip install --no-cache-dir --upgrade --upgrade-strategy only-if-needed -r "$d/requirements.txt" || true
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

# --- ComfyUI-Manager config: create/update from CM_* just before launch ---
configure_manager_config

# --- launch ComfyUI ---
COMFYUI_ARGS=""
if [ "${FORCE_SAGE_ATTENTION:-0}" = "1" ] && [ "${SAGE_ATTENTION_AVAILABLE:-0}" = "1" ]; then
    COMFYUI_ARGS="--use-sage-attention"
    log "Starting ComfyUI with SageAttention enabled"
elif [ "${FORCE_SAGE_ATTENTION:-0}" = "1" ]; then
    log "WARNING: FORCE_SAGE_ATTENTION=1 but SageAttention is not available; starting without it"
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
