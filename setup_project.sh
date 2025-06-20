#!/usr/bin/env bash
# ======================================================================
#  setup_project.sh  (venv + wheelhouse *offline* setup)
# ======================================================================

set -euo pipefail

# ----------------------------------------------------------------------
# 0. 変数
# ----------------------------------------------------------------------
PROJECT_ROOT="$(pwd)"
WHEEL_DIR="${PROJECT_ROOT}/wheelhouse"      # 事前 / 自動 DL 済み .whl の置き場
REQ_FILE="requirements.txt"
OUTPUT_DIR="midi_output"

VENV_DIR=".venv"
PYBIN="${VENV_DIR}/bin"
PIP="${PYBIN}/pip"
PYTHON="${PYBIN}/python"

# pip / build 系も wheelhouse で管理 (← wheel 必須! )
HEAVY_PACKAGES=(
  # --- core build ---
  "wheel>=0.43"            # ←★ 追加!!
  "pip>=24.0"
  "setuptools>=68.0"

  # --- numeric ---
  "numpy>=1.26.4,<2.0.0"   "scipy>=1.10"
  "scipy>=1.10"
  # --- parsing / util ---
  "PyYAML>=6.0"            "tomli>=2.0"
  "pydantic>=2.7"          "pydantic-core==2.33.2"
  # --- midi / audio ---
  "pretty_midi>=0.2.10"    "mido>=1.3.0"   "pydub>=0.25"
  # --- librosa & deps ---
  "soundfile>=0.12"        "audioread>=2.1.9"
  "numba>=0.57"            "llvmlite>=0.42" "librosa>=0.10"
  # --- plotting ---
  "matplotlib>=3.8" "contourpy>=1.0.1" \
  "fonttools>=4.22.0" "kiwisolver>=1.3.1" "Pillow>=10.0"
  # --- misc ---
  "charset_normalizer<4,>=2"
  # ── librosa & 依存 ─────────────
  "audioread>=2.1.9"
  "librosa>=0.10"
  # ── matplotlib (music21 連鎖) ─
  "kiwisolver>=1.3.1" "Pillow>=10.0"
)
# ----------------------------------------------------------------------
# 1. venv
# ----------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "🟢 0) create venv (${VENV_DIR})"
  python3 -m venv "${VENV_DIR}"
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
PYTAG="cp$(python3 - <<'PY'
import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')
PY
)"
for spec in "${HEAVY_PACKAGES[@]}"; do
  pkg="${spec%%[*<>=]*}"
  pattern="${WHEEL_DIR}/${pkg}"*-"${PYTAG}"*-manylinux*.whl
  if ls ${pattern} >/dev/null 2>&1; then continue; fi

  echo "   → fetch/build ${pkg}"
  if [[ "${pkg}" == "pretty_midi" ]]; then
    "${PYTHON}" -m pip wheel   --wheel-dir "${WHEEL_DIR}" --no-deps "${spec}"
  else
    "${PYTHON}" -m pip download --dest "${WHEEL_DIR}" \
      --platform manylinux_2_17_x86_64 --implementation cp --abi "${PYTAG}" \
      --only-binary=:all: --no-deps "${spec}"
  fi
done

# ----------------------------------------------------------------------
# 4. pip / setuptools / wheel を wheelhouse で更新
# ----------------------------------------------------------------------
echo "🟢 3) upgrade pip / setuptools / wheel (offline)"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" --upgrade pip setuptools wheel

# ----------------------------------------------------------------------
# 5. requirements.txt (プロジェクト依存) をオフラインで入れる
# ----------------------------------------------------------------------
echo "🟢 4) install project requirements"
"${PIP}" install --no-index --find-links="${WHEEL_DIR}" -r "${REQ_FILE}"

# ----------------------------------------------------------------------
# 6. プロジェクト自体を editable-install
# ----------------------------------------------------------------------
echo "🟢 5) install project (-e .)"
"${PIP}" install --no-build-isolation --no-deps -e .

# ----------------------------------------------------------------------
# 7. 後処理
# ----------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
echo "✅ setup finished!  run  'source ${VENV_DIR}/bin/activate'"
