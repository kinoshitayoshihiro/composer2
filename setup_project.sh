#!/usr/bin/env bash
# =====================================================================
#  setup_project.sh  (robust offline / online bootstrap)
# =====================================================================
set -euo pipefail

# ---------- paths ----------------------------------------------------
ROOT_DIR=$(pwd)
WHEEL_DIR="$ROOT_DIR/wheelhouse"
REQ_FILE="$ROOT_DIR/requirements.txt"
OUTPUT_DIR="$ROOT_DIR/midi_output"

VENV_DIR="$ROOT_DIR/.venv"
VPY="$VENV_DIR/bin/python"
VPIP="$VENV_DIR/bin/pip"

PYTAG="cp311"
MANYLINUX="manylinux2014_x86_64"

# ---------- heavy packages (1 token = 1 quoted string) ---------------
HEAVY_PKGS=(
  "wheel>=0.43" "pip>=24.0" "setuptools>=68.0"
  "numpy>=1.26.4,<2.0.0" "scipy>=1.10"
  "PyYAML>=6.0" "tomli>=2.0"
  "pydantic>=2.7" "pydantic-core==2.33.2"
  "pretty_midi>=0.2.10" "mido>=1.3.0" "pydub>=0.25"
  "soundfile>=0.12" "audioread>=2.1.9"
  "numba>=0.57" "llvmlite>=0.42" "librosa>=0.10"
  "matplotlib>=3.8" "contourpy>=1.0.1"
  "fonttools>=4.22.0" "kiwisolver>=1.3.1" "Pillow>=10.0"
  "charset_normalizer<4,>=2"
)

# ---------- 0. create / fix venv -------------------------------------
if [[ ! -x "$VPY" ]]; then
  echo "🟢 creating venv ($VENV_DIR)"
  python3 -m venv --symlinks "$VENV_DIR"   # ← ★ 失敗しないモード

  # フォールバック: python3 シンボリックリンクが無い環境用
  [[ -e "$VENV_DIR/bin/python3" ]] || ln -s python "$VENV_DIR/bin/python3"
fi
echo "   venv python: $("$VPY" -V)"

# ensure pip/setuptools/wheel は最低限入れる
"$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VPY" -m pip install --upgrade wheel pip setuptools >/dev/null

# ---------- 1. decide offline / online -------------------------------
if [[ ! -d "$WHEEL_DIR" || -z "$(ls -A "$WHEEL_DIR" 2>/dev/null)" ]]; then
  OFFLINE=false; mkdir -p "$WHEEL_DIR"
  echo "🟡 wheelhouse empty → ONLINE mode"
else
  OFFLINE=true
  echo "🟢 wheelhouse found → OFFLINE mode"
fi

# ---------- 2. fetch heavy wheels (only if online) -------------------
if ! $OFFLINE; then
  echo "🟢 downloading heavy wheels"
  for spec in "${HEAVY_PKGS[@]}"; do
    pkg=${spec%%[<>=]*}
    if compgen -G "$WHEEL_DIR/${pkg}-*-${PYTAG}-*manylinux*.whl" >/dev/null; then
      continue
    fi
    echo "   → $pkg"
    "$VPY" -m pip download --dest "$WHEEL_DIR" \
      --platform "$MANYLINUX" --implementation cp --abi "$PYTAG" \
      --python-version 3.11 --only-binary=:all: --no-deps "$spec" || true
  done
fi

# ---------- 3. upgrade pip etc. using wheelhouse ---------------------
"$VPIP" install --no-index --find-links "$WHEEL_DIR" \
  --upgrade wheel pip setuptools

# ---------- 4. requirements -----------------------------------------
echo "🟢 installing project requirements"
if $OFFLINE; then
  "$VPIP" install --no-index --find-links "$WHEEL_DIR" -r "$REQ_FILE"
else
  "$VPIP" install -r "$REQ_FILE"
fi

# ---------- 5. editable install project ------------------------------
"$VPIP" install --no-build-isolation --no-deps -e "$ROOT_DIR"

mkdir -p "$OUTPUT_DIR"
echo "✅ setup finished – run  'source $VENV_DIR/bin/activate'"
