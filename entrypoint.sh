#!/bin/bash
set -euo pipefail

APP_USER=${APP_USER:-appuser}
APP_GROUP=${APP_GROUP:-appuser}
PUID=${PUID:-1000}
PGID=${PGID:-1000}

BASE_DIR=/app/ComfyUI
CUSTOM_NODES_DIR="$BASE_DIR/custom_nodes"

# If running as root, map to requested UID/GID, fix ownership, and make Python install targets writable
if [ "$(id -u)" = "0" ]; then
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

  # Make Python system install targets writable for the runtime user
  # Includes: site-packages (purelib/platlib), scripts, headers, data, and data/share(+man1)
  # Discards anything not under /usr/local to avoid over-broad changes.
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
        print(v)  # stdout for the shell to capture
        log(f"emit {k} -> {v}")  # stderr for the user to see

d = paths.get("data")
if d:
    share = os.path.join(d, "share")
    man1 = os.path.join(share, "man", "man1")
    print(share)  # stdout
    print(man1)   # stdout
    log(f"emit wheel data dirs -> {share}, {man1}")  # stderr

log("Finished emitting target directories")
PY
)
  for d in "${PY_PATHS[@]}"; do
    case "$d" in
      /usr/local/*)
        mkdir -p "$d" || true
        chown -R "${APP_USER}:${APP_GROUP}" "$d" || true
        chmod -R u+rwX,g+rwX "$d" || true
        ;;
      *) : ;;
    esac
  done

  # Re-exec as the runtime user
  exec runuser -u "${APP_USER}" -- "$0" "$@"
fi

# Ensure ComfyUI-Manager exists (bind mounts can hide baked content)
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager" ]; then
  echo "[bootstrap] Installing ComfyUI-Manager into $CUSTOM_NODES_DIR/ComfyUI-Manager"
  git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git "$CUSTOM_NODES_DIR/ComfyUI-Manager" || true
fi

# User-site PATHs for --user installs (custom nodes)
export PATH="$HOME/.local/bin:$PATH"
pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="$HOME/.local/lib/python${pyver}/site-packages:${PYTHONPATH:-}"

# Auto-install custom node deps
if [ "${COMFY_AUTO_INSTALL:-1}" = "1" ]; then
  echo "[deps] Scanning custom nodes for requirements..."
  while IFS= read -r -d '' req; do
    echo "[deps] pip install --user -r $req"
    pip install --no-cache-dir --user -r "$req" || true
  done < <(find "$CUSTOM_NODES_DIR" -maxdepth 3 -type f \( -iname 'requirements.txt' -o -iname 'requirements-*.txt' -o -path '*/requirements/*.txt' \) -print0)

  while IFS= read -r -d '' pjt; do
    d="$(dirname "$pjt")"
    echo "[deps] pip install --user . in $d"
    (cd "$d" && pip install --no-cache-dir --user .) || true
  done < <(find "$CUSTOM_NODES_DIR" -maxdepth 2 -type f -iname 'pyproject.toml' -print0)

  pip check || true
fi

cd "$BASE_DIR"
exec "$@"
