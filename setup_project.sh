#!/usr/bin/env bash
# ======================================================================
#  setup_project.sh  (venv + wheelhouse *offline* setup)
# ======================================================================

set -euo pipefail

# ----------------------------------------------------------------------
# 0. 変数
# ----------------------------------------------------------------------
PROJECT_ROOT="$(pwd)"
WHEEL_DIR="${PROJECT_ROOT}/wheelhouse"
REQ_FILE="requirements.txt"
OUTPUT_DIR="midi_output"

VENV_DIR=".venv"
PYBIN="${VENV_DIR}/bin"
PIP="${PYBIN}/pip"
PYTHON="${PYBIN}/python"

# ターゲット Python / wheel タグ
PYTAG="cp311"                    # 例: Python 3.11
MANYLINUX_TAG="manylinux2014_x86_64"

# heavy パッケージ（wheel必須）
HEAVY_PACKAGES=(
  # build-core
  "wheel>=0.43" "pip>=24.0" "setuptools>=68.0"

  # numeric
  "numpy>=1.26.4,<2.0.0" "scipy>=1.10"

  # parsing / util
  "PyYAML>=6.0" "tomli>=2.0" "pydantic>=2.7" "pydantic-core==2.33.2"

  # midi / audio
  "pretty_midi>=0.2.10" "mido>=1.3.0" "pydub>=0.25"

  # librosa & deps
  "soundfile>=0.12" "audioread>=2.1.9" \
  "numba>=0.57" "llvmlite>=0.42" "librosa>=0.10"

  # plotting
  "matplotlib>=3.8" "contourpy>=1.0.1" \
  "fonttools>=4.22.0" "kiwisolver>=1.3.1" "Pillow>=10.0"

  # misc
  "charset_normalizer<4,>=2"
)

# ----------------------------------------------------------------------
# 1. venv
# ----------------------------------------------------------------------
if [[ ! -x "${PYTHON}" ]]; then
  echo "🟢 0) create venv (${VENV_DIR})"
  python3 -m venv "${VENV_DIR}" || {
    echo "❌ venv 作成に失敗"; exit 1; }
fi
echo "   venv Python: $(${PYTHON} -V)"

# ----------------------------------------------------------------------
# 2. wheelhouse 存在確認
# ----------------------------------------------------------------------
echo "🟢 1) check wheelhouse"
[[ -d "${WHEEL_DIR}" ]] || { echo "ERROR: '${WHEEL_DIR}' not found"; exit 1; }

# ----------------------------------------------------------------------
# 3. heavy packages を wheelhouse に補完 (オンライン時のみ)
# ----------------------------------------------------------------------
echo "🟢 2) ensure heavy wheels"
shopt -s nullglob       # glob 未ヒット時にそのまま文字列が残らない
for spec in "${HEAVY_PACKAGES[@]}"; do
  pkg="${spec%%[*<>=]*}"
  pattern=("${WHEEL_DIR}/${pkg}"*-"${PYTAG}"*-manylinux*.whl)
  if (( ${#pattern[@]} )); then continue; fi

  echo "   → fetch/build ${pkg}"
  if [[ "${pkg}" == "pretty_midi" ]]; then
    "${PYTHON}" -m pip wheel --wheel-dir "${WHEEL_DIR}" --no-deps "${spec}"
  else
    "${PYTHON}" -m pip download --dest "${WHEEL_DIR}" \
      --platform "${MANYLINUX_TAG}" \
      --implementation cp --abi "${PYTAG}" \
      --python-version "${PYTAG:2}" \
      --only-binary=:all: --no-deps "${spec}"
  fi
done
shopt -u nullglob

# ----------------------------------------------------------------------
# 4. build-tool trio を wheelhouse から install / upgrade
# ----------------------------------------------------------------------
echo "🟢 3) upgrade pip / setuptools / wheel (offline)"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" --upgrade pip setuptools wheel

# ----------------------------------------------------------------------
# 5. requirements.txt をオフライン install
# ----------------------------------------------------------------------
echo "🟢 4) install project requirements"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" -r "${REQ_FILE}"

# ----------------------------------------------------------------------
# 6. プロジェクト自体を editable install
# ----------------------------------------------------------------------
echo "🟢 5) install project (-e .)"
"${PIP}" install --no-build-isolation --no-deps -e .

# ----------------------------------------------------------------------
# 7. 後処理
# ----------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
echo "✅ setup finished!   source ${VENV_DIR}/bin/activate"
