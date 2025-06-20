#!/usr/bin/env bash
set -euo pipefail

# -------------------------------- constants
WHEEL_DIR="wheelhouse"
REQ_FILE="requirements.txt"
VENV_DIR=".venv"
PYBIN="${VENV_DIR}/bin"
PYTAG="$($(command -v python3) - <<'PY' 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")' PY)"

# --- heavy / ビルドが遅いパッケージ ------------
HEAVY_PACKAGES=(
  # ── Core ────────────────────────────
  "numpy>=1.26.4,<2.0.0"
  "scipy>=1.10"
  "pydantic>=2.7"            "pydantic-core==2.33.2"
  "PyYAML>=6.0"              "tomli>=2.0"
  "pretty_midi>=0.2.10"      "mido>=1.3.0"       "pydub>=0.25"
  # ── librosa + audio ──────────────────
  "soundfile>=0.12"          "audioread>=2.1.9"
  "numba>=0.57"              "llvmlite>=0.42"
  "librosa>=0.10"
  # ── matplotlib stack (music21) ───────
  "matplotlib>=3.8" "contourpy>=1.0.1" "fonttools>=4.22.0" \
  "kiwisolver>=1.3.1" "Pillow>=10.0"
)

# -------------------------------- venv
if [[ ! -d "${VENV_DIR}" ]]; then python3 -m venv "${VENV_DIR}"; fi
PIP="${PYBIN}/pip"; PYTHON="${PYBIN}/python"

# -------------------------------- wheel fetch
for spec in "${HEAVY_PACKAGES[@]}"; do
  pkg="${spec%%[*<>=]*}"
  pattern="${WHEEL_DIR}/${pkg}"*-${PYTAG}-*-manylinux*.whl

  [[ -e ${pattern} ]] && continue

  echo "• ${pkg} – wheel missing, trying download/build …"
  if "${PIP}" download --dest "${WHEEL_DIR}" \
        --only-binary=:all: --no-deps --platform manylinux_2_17_x86_64 \
        --implementation cp --abi "${PYTAG}" "${spec}" 2>/dev/null; then
        continue
  fi
  echo "  ↳ building from source"
  "${PIP}" wheel --wheel-dir "${WHEEL_DIR}" --no-deps "${spec}"
done

# -------------------------------- install
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" --upgrade pip setuptools
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" \
  --upgrade-strategy only-if-needed -r "${REQ_FILE}"
"${PIP}" install --no-build-isolation --no-deps -e .

echo "✅ setup finished"
