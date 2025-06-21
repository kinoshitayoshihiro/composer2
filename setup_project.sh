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
PYTHON="${PYBIN}/python"
PIP="${PYBIN}/pip"

# Wheel 名に使うタグ
PYTAG="cp311"                         # ★ コンテナ側 Python3.11 系想定
MANYLINUX_TAG="manylinux_2_17_x86_64" # ★ ほぼ全 wheel が用意されている基準

# ----------------------------------------------------------------------
# 1. venv
# ----------------------------------------------------------------------
if [[ ! -x "${PYTHON}" ]]; then
  echo "🟢 0) create venv (${VENV_DIR})"
  python3 -m venv --copies "${VENV_DIR}" \
    || { echo "❌ venv 作成に失敗"; exit 1; }
fi
echo "   venv Python: $(${PYTHON} -V)"

# ----------------------------------------------------------------------
# 2. heavy パッケージ一覧
#   ※ wheel 必須 or ビルドが重い物。被りは OK（--exists-action i）
# ----------------------------------------------------------------------
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
# 3. wheelhouse 存在確認
# ----------------------------------------------------------------------
echo "🟢 1) check wheelhouse"
[[ -d "${WHEEL_DIR}" ]] || { echo "ERROR: '${WHEEL_DIR}' not found"; exit 1; }

# ----------------------------------------------------------------------
# 4. heavy wheels を補完（オンライン時のみ）
# ----------------------------------------------------------------------
echo "🟢 2) ensure heavy wheels"
for spec in "${HEAVY_PACKAGES[@]}"; do
  pkg="${spec%%[*<>=]*}"
  pattern="${WHEEL_DIR}/${pkg}"*-"${PYTAG}"*-manylinux*.whl

  # 既に wheel があればスキップ
  if ls ${pattern} >/dev/null 2>&1; then
    continue
  fi

  echo "   → ${pkg}"
  if [[ "${pkg}" == "pretty_midi" ]]; then
    # pretty_midi はソースビルド (依存 small)
    "${PYTHON}" -m pip wheel --wheel-dir "${WHEEL_DIR}" --no-deps "${spec}"
  else
    # manylinux wheel を取得
    "${PYTHON}" -m pip download \
      --dest "${WHEEL_DIR}" \
      --platform "${MANYLINUX_TAG}" \
      --implementation cp --abi "${PYTAG}" \
      --only-binary=:all: --no-deps "${spec}"
  fi
done

# ----------------------------------------------------------------------
# 5. pip / setuptools / wheel を wheelhouse で更新
# ----------------------------------------------------------------------
echo "🟢 3) upgrade pip / setuptools / wheel (offline)"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" --upgrade pip setuptools wheel

# ----------------------------------------------------------------------
# 6. requirements.txt をオフラインインストール
# ----------------------------------------------------------------------
echo "🟢 4) install project requirements (offline)"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" -r "${REQ_FILE}"

# ----------------------------------------------------------------------
# 7. プロジェクト自体を editable-install
# ----------------------------------------------------------------------
echo "🟢 5) install project (-e .)"
"${PIP}" install --no-build-isolation --no-deps -e .

# ----------------------------------------------------------------------
# 8. 後処理
# ----------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
echo "✅ setup finished!  run  'source ${VENV_DIR}/bin/activate'"
