#!/bin/bash
set -euo pipefail

# --- config ---
APP_USER=${APP_USER:-appuser}
APP_GROUP=${APP_GROUP:-appuser}
PUID=${PUID:-1000}
PGID=${PGID:-1000}
BASE_DIR=/app/ComfyUI
CUSTOM_NODES_DIR="$BASE_DIR/custom_nodes"
PERMISSIONS_SET_FLAG="$BASE_DIR/.permissions_set"
FIRST_RUN_FLAG="$BASE_DIR/.first_run_done"

# Manager config (persistent)
USER_DIR="$BASE_DIR/user"
CM_CFG_DIR="$USER_DIR/default/ComfyUI-Manager"
CM_CFG="$CM_CFG_DIR/config.ini"
CM_SEEDED_FLAG="$CM_CFG_DIR/.config_seeded"

# --- logging ---
log() { echo "[$(date '+%H:%M:%S')] $1"; }

# Make newly created files group-writable (helps in shared volumes)
umask 0002

# --- ComfyUI-Manager config from CM_* env ---
configure_manager_config() {
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

    exec runuser -p -u "${APP_USER}" -- "$0" "$@"
fi

# --- From here on, running as $APP_USER ---

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
log "Starting ComfyUI..."
cd "$BASE_DIR"
if [ $# -eq 0 ]; then
    exec python main.py --listen 0.0.0.0
else
    if [ "$1" = "python" ] && [ "${2:-}" = "main.py" ]; then
        shift 2; exec python main.py "$@"
    else
        exec "$@"
    fi
fi
