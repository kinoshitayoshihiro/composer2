#!/usr/bin/env bash
# ======================================================================
#  setup_project.sh      (venv 付き・オフライン wheelhouse セットアップ)
# ======================================================================
set -euo pipefail

# ----------------------------------------------------------------------
#  0. 定数
# ----------------------------------------------------------------------
PROJECT_ROOT="$(pwd)"
WHEEL_DIR="${PROJECT_ROOT}/wheelhouse"      # 事前に wheel を集める場所
REQ_FILE="requirements.txt"                 # 通常の requirements
OUTPUT_DIR="midi_output"                    # 生成ファイル格納先
VENV_DIR=".venv"                            # venv を置くフォルダ
PYBIN="${VENV_DIR}/bin"

# Python tag (cp310 / cp311 など) を動的取得
PYTAG="$((command -v python3) - <<'PY' 
import sys
print(f'cp{sys.version_info.major}{sys.version_info.minor}')
PY
)"

# ----------------------------------------------------------------------
#  1. heavy / バイナリ依存が大きいパッケージ一覧
#     → wheelhouse に必ず置く
# ----------------------------------------------------------------------
HEAVY_PACKAGES=(
  # ── Core numeric ───────────────
  "numpy>=1.26.4,<2.0.0"
  "scipy>=1.10"
  # ── Validation / parsing ───────
  "pydantic>=2.7"          "pydantic-core==2.33.2"
  "PyYAML>=6.0"            "tomli>=2.0"
  # ── MIDI / audio utils ─────────
  "pretty_midi>=0.2.10"    "mido>=1.3.0"     "pydub>=0.25"
  # ── librosa & 依存 ─────────────
  "soundfile>=0.12"
  "audioread>=2.1.9"
  "numba>=0.57"            "llvmlite>=0.42"
  "librosa>=0.10"
  # ── matplotlib (music21 連鎖) ─
  "matplotlib>=3.8" "contourpy>=1.0.1" "fonttools>=4.22.0" \
  "kiwisolver>=1.3.1" "Pillow>=10.0"
)

# ----------------------------------------------------------------------
#  2. venv 作成
# ----------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  echo ">> Creating venv in ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

PIP="${PYBIN}/pip"
PYTHON="${PYBIN}/python"
echo ">> Using interpreter: $(${PYTHON} -V)"

# ----------------------------------------------------------------------
#  3. wheelhouse 存在チェック
# ----------------------------------------------------------------------
if [[ ! -d "${WHEEL_DIR}" ]]; then
  echo "ERROR: wheelhouse '${WHEEL_DIR}' がありません。" >&2
  exit 1
fi

# ----------------------------------------------------------------------
#  4. heavy パッケージの wheel を用意
# ----------------------------------------------------------------------
echo ">> Ensuring heavy wheels ..."
for spec in "${HEAVY_PACKAGES[@]}"; do
  pkg="${spec%%[<=>]*}"
  pattern="${WHEEL_DIR}/${pkg}"*-${PYTAG}-*-manylinux*.whl

  # 既に wheel が有ればスキップ
  if ls ${pattern} >/dev/null 2>&1; then
    continue
  fi

  echo "   • ${pkg} – trying download..."
  if "${PIP}" download --dest "${WHEEL_DIR}" \
        --only-binary=:all: --no-deps \
        --platform manylinux_2_17_x86_64 \
        --implementation cp --abi "${PYTAG}" "${spec}" 2>/dev/null; then
        continue
  fi

  echo "     ↳ no ready-made wheel; building locally"
  "${PIP}" wheel --wheel-dir "${WHEEL_DIR}" --no-deps "${spec}"
done

# ----------------------------------------------------------------------
#  5. pip / setuptools upgrade (オフライン)
# ----------------------------------------------------------------------
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" --upgrade pip setuptools wheel

# ----------------------------------------------------------------------
#  6. requirements.txt インストール
#     --upgrade-strategy only-if-needed で既存 wheel 優先
# ----------------------------------------------------------------------
echo ">> Installing requirements.txt ..."
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" \
  --upgrade-strategy only-if-needed -r "${REQ_FILE}"

# ----------------------------------------------------------------------
#  7. プロジェクト本体を editable インストール
# ----------------------------------------------------------------------
echo ">> Installing project (editable) ..."
"${PIP}" install --no-build-isolation --no-deps -e .

# ----------------------------------------------------------------------
#  8. 後処理
# ----------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
echo "✅ Setup finished; run 'source ${VENV_DIR}/bin/activate' to enter the environment."
