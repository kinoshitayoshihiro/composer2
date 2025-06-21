#!/usr/bin/env bash
# =====================================================================
#  setup_project.sh  (venv + wheelhouse *offline* setup)
# =====================================================================
set -euo pipefail

# ---------------------------------------------------------------------
# 0. paths / vars
# ---------------------------------------------------------------------
PROJECT_ROOT="$(pwd)"
WHEEL_DIR="${PROJECT_ROOT}/wheelhouse"        # 事前 DL した .whl 置き場
REQ_FILE="requirements.txt"
OUTPUT_DIR="midi_output"

VENV_DIR=".venv"
PYBIN="${VENV_DIR}/bin"
PYTHON="${PYBIN}/python"
PIP="${PYBIN}/pip"

# Python tags (host≒GitHub Actions: cp 3.11 / manylinux2014)
PYTAG="cp311"
MANYLINUX_TAG="manylinux2014_x86_64"

# ---------------------------------------------------------------------
# 1. heavy package list  ※-d '' を削除しヒアドキュメントで閉じる
# ---------------------------------------------------------------------
read -r -d '' HEAVY_SPEC <<'PKG'
wheel>=0.43
pip>=24.0
setuptools>=68.0
numpy>=1.26.4,<2.0.0
scipy>=1.10
PyYAML>=6.0
tomli>=2.0
pydantic>=2.7
pydantic-core==2.33.2
pretty_midi>=0.2.10
mido>=1.3.0
pydub>=0.25
soundfile>=0.12
audioread>=2.1.9
numba>=0.57
llvmlite>=0.42
librosa>=0.10
matplotlib>=3.8
contourpy>=1.0.1
fonttools>=4.22.0
kiwisolver>=1.3.1
Pillow>=10.0
charset_normalizer<4,>=2
PKG
mapfile -t HEAVY_PACKAGES <<<"${HEAVY_SPEC}"

# ---------------------------------------------------------------------
# 2. create venv (–copies で bin/python3 不足を防ぐ)
# ---------------------------------------------------------------------
if [[ ! -x "${PYTHON}" ]]; then
  echo "🟢 0) create venv (${VENV_DIR})"
  python3 -m venv --copies "${VENV_DIR}"
fi
echo "   venv Python: $(${PYTHON} -V)"

# ---------------------------------------------------------------------
# 3. check wheelhouse
# ---------------------------------------------------------------------
echo "🟢 1) check wheelhouse"; [[ -d "${WHEEL_DIR}" ]] \
  || { echo "ERROR: '${WHEEL_DIR}' not found"; exit 1; }

# ---------------------------------------------------------------------
# 4. fetch / build heavy wheels (オンライン時のみ)
# ---------------------------------------------------------------------
echo "🟢 2) ensure heavy wheels"
for spec in "${HEAVY_PACKAGES[@]}"; do
  pkg="${spec%%[*<>=]*}"
  pattern="${WHEEL_DIR}/${pkg}"*-${PYTAG}-*manylinux*.whl
  [[ -e $(ls ${pattern} 2>/dev/null | head -n1) ]] && continue

  echo "   → ${pkg}"
  if [[ "${pkg}" == "pretty_midi" ]]; then
    "${PYTHON}" -m pip wheel --wheel-dir "${WHEEL_DIR}" --no-deps "${spec}"
  else
    "${PYTHON}" -m pip download --dest "${WHEEL_DIR}" \
      --platform "${MANYLINUX_TAG}" --implementation cp --abi "${PYTAG}" \
      --python-version 3.11 --only-binary=:all: --no-deps "${spec}"
  fi
done

# ---------------------------------------------------------------------
# 5. upgrade pip / setuptools / wheel  (offline)
# ---------------------------------------------------------------------
echo "🟢 3) upgrade pip / setuptools / wheel"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" \
  --upgrade pip setuptools wheel

# ---------------------------------------------------------------------
# 6. install project requirements (offline)
# ---------------------------------------------------------------------
echo "🟢 4) install requirements"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" -r "${REQ_FILE}"

# ---------------------------------------------------------------------
# 7. install project itself
# ---------------------------------------------------------------------
echo "🟢 5) install project (-e .)"
"${PIP}" install --no-build-isolation --no-deps -e .

# ---------------------------------------------------------------------
# 8. post-setup
# ---------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
echo "✅ setup finished – run  'source ${VENV_DIR}/bin/activate'"
