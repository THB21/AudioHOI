#!/usr/bin/env bash
# Download the license-gated SMPL-X (and optionally SMPL) body models from MPI and
# place them where the AudioHOI generic pipeline expects them.
#
#   scripts/third-party/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
#   scripts/third-party/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl   (optional)
#
# RUN THIS IN YOUR OWN SSH SESSION (not via the assistant), so your MPI credentials
# are typed with a hidden prompt and never leave your machine.
#
#   bash scripts/setup_body_models.sh
#
# Requires a registered account at https://smpl-x.is.tue.mpg.de (and https://smpl.is.tue.mpg.de
# for the optional SMPL model). If the automated login fails (MPI occasionally changes the
# endpoint), just download the archives in a browser and re-run with LOCAL_SMPLX_ZIP / LOCAL_SMPL_ZIP
# pointing at them — see the bottom of this script.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/scripts/third-party/GVHMR/inputs/checkpoints/body_models"
SMPLX_DIR="$DEST/smplx"
SMPL_DIR="$DEST/smpl"
mkdir -p "$SMPLX_DIR" "$SMPL_DIR"
TMP="$(mktemp -d)"
COOKIE="$TMP/mpi.cookie"
trap 'rm -rf "$TMP"' EXIT

echo "=== MPI credentials (used only for this download; not stored) ==="
read -rp  "MPI email/username: " MPI_USER
read -rsp "MPI password:       " MPI_PASS; echo

login() {
  : # Auth happens inline in fetch() via a POST to download.php (official MPI flow).
}

fetch() {  # fetch <domain> <sfile> <out.zip>
  local domain="$1" sfile="$2" out="$3"
  echo "  downloading $sfile ..."
  # Official MPI pattern: POST url-encoded credentials directly to download.php.
  # (--data-urlencode is essential — passwords often contain & ! etc.)
  curl -L -s --cookie-jar "$COOKIE" \
       "https://download.is.tue.mpg.de/download.php?domain=${domain}&sfile=${sfile}&resume=1" \
       --data-urlencode "username=$MPI_USER" \
       --data-urlencode "password=$MPI_PASS" \
       -o "$out"
  # A failed auth returns an HTML login page, not a zip. Validate the magic bytes.
  if ! unzip -tq "$out" >/dev/null 2>&1; then
    echo "  !! $sfile did not download as a valid zip (login likely failed)." >&2
    echo "     Head of the file:" >&2; head -c 200 "$out" >&2; echo >&2
    return 1
  fi
}

# ---------------- SMPL-X (required) ----------------
SMPLX_ZIP="${LOCAL_SMPLX_ZIP:-$TMP/models_smplx_v1_1.zip}"
if [ -z "${LOCAL_SMPLX_ZIP:-}" ]; then
  login
  fetch "smplx" "models_smplx_v1_1.zip" "$SMPLX_ZIP"
fi
echo "=== extracting SMPL-X ==="
unzip -o -q "$SMPLX_ZIP" -d "$TMP/smplx_extract"
# Archive layout: models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz
find "$TMP/smplx_extract" -iname "SMPLX_*.npz" -exec cp -v {} "$SMPLX_DIR/" \;

# ---------------- SMPL (optional, for full-body SMPL rendering/eval) ----------------
if [ "${WITH_SMPL:-0}" = "1" ]; then
  SMPL_ZIP="${LOCAL_SMPL_ZIP:-$TMP/SMPL_python_v.1.1.0.zip}"
  if [ -z "${LOCAL_SMPL_ZIP:-}" ]; then
    login
    fetch "smpl" "SMPL_python_v.1.1.0.zip" "$SMPL_ZIP"
  fi
  echo "=== extracting SMPL ==="
  unzip -o -q "$SMPL_ZIP" -d "$TMP/smpl_extract"
  # basic model pkls live under SMPL_python_v.1.1.0/smpl/models/basicmodel_*.pkl
  # GVHMR expects SMPL_{GENDER}.pkl
  declare -A MAP=( [NEUTRAL]="neutral" [MALE]="m" [FEMALE]="f" )
  for G in NEUTRAL MALE FEMALE; do
    src=$(find "$TMP/smpl_extract" -iname "*${MAP[$G]}*.pkl" | head -1 || true)
    [ -n "$src" ] && cp -v "$src" "$SMPL_DIR/SMPL_${G}.pkl"
  done
fi

echo ""
echo "=== DONE. Installed body models: ==="
ls -la "$SMPLX_DIR" "$SMPL_DIR"
echo ""
echo "Sanity: the pipeline needs at minimum  $SMPLX_DIR/SMPLX_NEUTRAL.npz"
[ -f "$SMPLX_DIR/SMPLX_NEUTRAL.npz" ] && echo "OK: SMPLX_NEUTRAL.npz present" || echo "MISSING: SMPLX_NEUTRAL.npz"

# ------------------------------------------------------------------
# Browser fallback (if automated MPI login fails):
#   1. Log in at https://smpl-x.is.tue.mpg.de  -> Downloads -> "SMPL-X v1.1 (NPZ+PKL, 830 MB)"
#      = models_smplx_v1_1.zip
#   2. scp it to this box, then:  LOCAL_SMPLX_ZIP=/path/to/models_smplx_v1_1.zip bash scripts/setup_body_models.sh
#   (For SMPL too: add WITH_SMPL=1 and LOCAL_SMPL_ZIP=/path/to/SMPL_python_v.1.1.0.zip)
# ------------------------------------------------------------------
