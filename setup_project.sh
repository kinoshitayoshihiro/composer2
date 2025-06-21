#!/usr/bin/env bash
# ======================================================================
#  setup_project.sh  (venv + wheelhouse *offline* setup)
# ======================================================================

set -euo pipefail

# ----------------------------------------------------------------------
# 変数
# ----------------------------------------------------------------------
PROJECT_ROOT="$(pwd)"
WHEEL_DIR="${PROJECT_ROOT}/wheelhouse"
REQ_FILE="requirements.txt"
OUTPUT_DIR="midi_output"

VENV_DIR=".venv"
PYBIN="${VENV_DIR}/bin"
# まだ venv が無い段階なので PYTHON/PIP は後で改めて設定

PYTAG="cp311"
MANYLINUX_TAG="manylinux_2_17_x86_64"

# heavy-list はヒアドキュメントのまま
read -r -d '' HEAVY_SPEC <<'LIST'
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
LIST

mapfile -t HEAVY_PACKAGES <<<"${HEAVY_SPEC}"

# ----------------------------------------------------------------------
# 0. venv
# ----------------------------------------------------------------------
if [[ ! -e "${VENV_DIR}/pyvenv.cfg" ]]; then
  echo "🟢 0) create venv (${VENV_DIR})"
  python3 -m venv "${VENV_DIR}"   # ← --copies を外す
fi

# venv 内の python / pip を改めて取得
PYTHON="${PYBIN}/python"
PIP="${PYBIN}/pip"

if [[ ! -x "${PYTHON}" ]]; then
  echo "❌ venv 作成に失敗: ${PYTHON} がありません" >&2
  exit 1
fi
echo "   venv Python: $(${PYTHON} -V)"

# ----------------------------------------------------------------------
echo "🟢 1) check wheelhouse"
[[ -d "${WHEEL_DIR}" ]] || { echo "ERROR: '${WHEEL_DIR}' not found"; exit 1; }

# ----------------------------------------------------------------------
echo "🟢 2) ensure heavy wheels"
for spec in "${HEAVY_PACKAGES[@]}"; do
  pkg="${spec%%[*<>=]*}"
  pattern="${WHEEL_DIR}/${pkg}"*-"${PYTAG}"*-manylinux*.whl
  [[ -e $(echo "${pattern}") ]] && continue

  echo "   → ${pkg}"
  if [[ "${pkg}" == "pretty_midi" ]]; then
    "${PYTHON}" -m pip wheel --wheel-dir "${WHEEL_DIR}" --no-deps "${spec}"
  else
    "${PYTHON}" -m pip download \
      --dest "${WHEEL_DIR}" \
      --platform "${MANYLINUX_TAG}" \
      --implementation cp --abi "${PYTAG}" \
      --only-binary=:all: --no-deps "${spec}"
  fi
done

# ----------------------------------------------------------------------
echo "🟢 3) upgrade pip / setuptools / wheel (offline)"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" --upgrade pip setuptools wheel

# ----------------------------------------------------------------------
echo "🟢 4) install project requirements (offline)"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" -r "${REQ_FILE}"

# ----------------------------------------------------------------------
echo "🟢 5) install project (-e .)"
"${PIP}" install --no-build-isolation --no-deps -e .

# ----------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
echo "✅ setup finished!   source ${VENV_DIR}/bin/activate"
