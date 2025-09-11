#!/bin/bash
set -Eeuo pipefail

# Timestamped xtrace (enable by uncommenting: set -x)
export PS4='+ [${EPOCHREALTIME}] ${BASH_SOURCE##*/}:${LINENO}: '
# set -x

log() { printf '[entrypoint] %s\n' "$*" >&2; }

APP_USER=${APP_USER:-appuser}
APP_GROUP=${APP_GROUP:-appuser}
PUID=${PUID:-1000}
PGID=${PGID:-1000}
BASE_DIR=/app/ComfyUI
CUSTOM_NODES_DIR="$BASE_DIR/custom_nodes"

if [ "$(id -u)" = "0" ]; then
  log "Running as root; mapping to UID=${PUID} GID=${PGID} and fixing ownerships" 
  # Map group to PGID if it already exists, otherwise remap the named group
  if getent group "${PGID}" >/dev/null; then
    EXISTING_GRP="$(getent group "${PGID}" | cut -d: -f1)"
    log "Using existing group ${EXISTING_GRP} for GID ${PGID}"
    usermod -g "${EXISTING_GRP}" "${APP_USER}" || true
    APP_GROUP="${EXISTING_GRP}"
  else
    log "Remapping group ${APP_GROUP} to GID ${PGID}"
    groupmod -o -g "${PGID}" "${APP_GROUP}" || true
  fi

  # Map user to PUID
  log "Remapping user ${APP_USER} to UID ${PUID}"
  usermod -o -u "${PUID}" "${APP_USER}" || true

  # Ensure home and app dir exist and are owned
  mkdir -p "/home/${APP_USER}"
  for d in "$BASE_DIR" "/home/$APP_USER"; do
    if [ -e "$d" ]; then
      log "Ensuring ownership ${APP_USER}:${APP_GROUP} on ${d}"
      chown -R "${APP_USER}:${APP_GROUP}" "$d" || true
    fi
  done

  # Discover Python system install targets and make them writable for runtime user
  # Includes: site-packages (purelib/platlib), scripts, headers, data, and data/share(+man1)
  # Discards anything not under /usr/local to avoid over-broad changes.
  log "Discovering Python install paths via sysconfig (unbuffered Python logging enabled)"
  readarray -t PY_PATHS < <(
    env PYTHONUNBUFFERED=1 python - <<'PY'
import sys, os, sysconfig

def emit(label, path):
    if not path:
        return
    # stdout: raw path for readarray
    print(path)
    # stderr: labeled diagnostics for logs
    print(f"[py] {label}: {path}", file=sys.stderr, flush=True)

p = sysconfig.get_paths()
for k in ("purelib","platlib","scripts","include","platinclude","data"):
    emit(k, p.get(k))

d = p.get("data")
if d:
    # Wheel .data/data payloads commonly land under {data}/share, including manpages like ttx.1
    emit("share", os.path.join(d, "share"))
    emit("man1", os.path.join(d, "share", "man", "man1"))
PY
  )

  log "Found ${#PY_PATHS[@]} Python target paths:"
  for p in "${PY_PATHS[@]}"; do
    printf '  - %s\n' "$p" >&2
  done

  for d in "${PY_PATHS[@]}"; do
    case "$d" in
      /usr/local/*)
        log "Ensuring writable under /usr/local: ${d}"
        mkdir -p "$d" || true
        chown -R "${APP_USER}:${APP_GROUP}" "$d" || true
        chmod -R u+rwX,g+rwX "$d" || true
        ;;
      *)
        log "Skipping non-/usr/local path: ${d}"
        ;;
    esac
  done

  log "Re-exec as runtime user ${APP_USER} (UID=${PUID}, GID=${PGID})"
  exec runuser -u "${APP_USER}" -- "$0" "$@"
fi

log "Running as $(id -un) (UID=$(id -u))"

# Ensure ComfyUI-Manager exists (bind mounts can hide baked content)
if [ ! -d "$CUSTOM_NODES_DIR/ComfyUI-Manager" ]; then
  log "Installing ComfyUI-Manager into $CUSTOM_NODES_DIR/ComfyUI-Manager"
  git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git "$CUSTOM_NODES_DIR/ComfyUI-Manager" || true
fi

# User-site PATHs for --user installs (custom nodes)
export PATH="$HOME/.local/bin:$PATH"
pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONPATH="$HOME/.local/lib/python${pyver}/site-packages:${PYTHONPATH:-}"

# Auto-install custom node deps
if [ "${COMFY_AUTO_INSTALL:-1}" = "1" ]; then
  log "[deps] Scanning custom nodes for requirements..."
  while IFS= read -r -d '' req; do
    log "[deps] pip install --user -r $req"
    pip install --no-cache-dir --user -r "$req" || true
  done < <(find "$CUSTOM_NODES_DIR" -maxdepth 3 -type f \( -iname 'requirements.txt' -o -iname 'requirements-*.txt' -o -path '*/requirements/*.txt' \) -print0)

  while IFS= read -r -d '' pjt; do
    d="$(dirname "$pjt")"
    log "[deps] pip install --user . in $d"
    (cd "$d" && pip install --no-cache-dir --user .) || true
  done < <(find "$CUSTOM_NODES_DIR" -maxdepth 2 -type f -iname 'pyproject.toml' -print0)

  log "[deps] pip check (sanity)"
  pip check || true
fi

cd "$BASE_DIR"
exec "$@"
