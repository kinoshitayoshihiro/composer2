#!/usr/bin/env bash
# =========================================================
# setup_project.sh   (robust offline / online bootstrap)
# =========================================================
set -euo pipefail

# ---------------------------------------------------------
# 0. Paths / variables
# ---------------------------------------------------------
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="${ROOT_DIR}/wheelhouse"
REQ_FILE="${ROOT_DIR}/requirements.txt"
OUTPUT_DIR="${ROOT_DIR}/midi_output"

VENV_DIR="${ROOT_DIR}/.venv"
PYBIN="${VENV_DIR}/bin"
VPY="${PYBIN}/python"          # venv-local python
VPIP="${PYBIN}/pip"

# 環境 (ビルドタグは *ダウンロード側* にしか要らない)
PYTAG="cp311"
MANYLINUX_TAG="manylinux2014_x86_64"

# 重量級パッケージ（wheel を確保しておきたい物）
HEAVY_PKGS=(
  wheel>=0.43 pip>=24 setuptools>=68
  numpy>=1.26.4,<2 scipy>=1.10  PyYAML>=6 tomli>=2
  pydantic>=2.7 pydantic-core==2.33.2
  pretty_midi>=0.2.10 mido>=1.3.0 pydub>=0.25
  soundfile>=0.12 audioread>=2.1.9
  numba>=0.57 llvmlite>=0.42 librosa>=0.10
  matplotlib>=3.8 contourpy>=1 fonttools>=4.22
  kiwisolver>=1.3.1 Pillow>=10 charset_normalizer<4,>=2
)

# ---------------------------------------------------------
# 1. Create venv *once* (use whichever python is on PATH)
# ---------------------------------------------------------
if [[ ! -x "${VPY}" ]]; then
  echo "🟢 creating venv (${VENV_DIR})"
  python3 -m venv "${VENV_DIR}"
fi
echo "   venv python: $("${VPY}" -V)"

# ---------------------------------------------------------
# 2. Decide offline / online mode
#    (wheelhouse 目录が空ならオンラインで取得)
# ---------------------------------------------------------
OFFLINE=true
if [[ ! -d "${WHEEL_DIR}" ]] || [[ -z $(ls -A "${WHEEL_DIR}" 2>/dev/null) ]]; then
  echo "🟡 wheelhouse missing / empty → ONLINE mode"
  OFFLINE=false
else
  echo "🟢 wheelhouse found → OFFLINE mode"
fi

# ---------------------------------------------------------
# 3. If online, pre-fetch wheels for heavy packages
# ---------------------------------------------------------
if ! $OFFLINE; then
  mkdir -p "${WHEEL_DIR}"
  echo "🟢 downloading heavy wheels"
  for spec in "${HEAVY_PKGS[@]}"; do
    pkg=${spec%%[<>=]*}
    if compgen -G "${WHEEL_DIR}/${pkg}-*-${PYTAG}-*manylinux*.whl" >/dev/null; then
      continue
    fi
    echo "   → ${pkg}"
    "${VPY}" -m pip download --dest "${WHEEL_DIR}" \
      --platform "${MANYLINUX_TAG}" --implementation cp --abi "${PYTAG}" \
      --python-version 3.11 --only-binary=:all: --no-deps "${spec}" || true
  done
fi

# ---------------------------------------------------------
# 4. Bootstrap pip / wheel INSIDE the venv
#    （wheel がまだ無い状態に対応）
# ---------------------------------------------------------
# ensurepip が無いディストリもあるため fallback
if ! "${VPY}" -m pip --version >/dev/null 2>&1; then
  "${VPY}" -m ensurepip --upgrade
fi

# オフラインなら wheelhouse 参照、オンラインなら普通にアップグレード
if $OFFLINE; then
  "${VPY}" -m pip install --no-index --find-links "${WHEEL_DIR}" \
    --upgrade wheel pip setuptools
else
  "${VPY}" -m pip install --upgrade wheel pip setuptools
fi

# ---------------------------------------------------------
# 5. Install project requirements
# ---------------------------------------------------------
if $OFFLINE; then
  "${VPIP}" install --no-index --find-links "${WHEEL_DIR}" -r "${REQ_FILE}"
else
  "${VPIP}" install -r "${REQ_FILE}"
fi

# ---------------------------------------------------------
# 6. Editable-install the project itself
# ---------------------------------------------------------
"${VPIP}" install --no-build-isolation --no-deps -e "${ROOT_DIR}"

# ---------------------------------------------------------
# 7. Misc
# ---------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
echo "✅ setup finished – run  'source ${VENV_DIR}/bin/activate'"
