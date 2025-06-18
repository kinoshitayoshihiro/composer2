#!/usr/bin/env bash
# ======================================================================
#  setup_project.sh
#  ・wheelhouse 内のホイールを最優先に、完全オフラインで依存を解決
#  ・欠けている heavy パッケージはオンライン時に自動取得／ビルド
#  ・プロジェクトを editable (-e) でインストール
# ======================================================================

set -euo pipefail

# ----------------------------------------------------------------------
#  定数
# ----------------------------------------------------------------------
WHEEL_DIR="./wheelhouse"
REQ_FILE="requirements.txt"
OUTPUT_DIR="midi_output"
PROJECT_ROOT="$(pwd)"

# 必ずホイールで揃えたいパッケージ一覧
HEAVY_PACKAGES=(
  # requirements.txt 直読み
  "scipy>=1.10"
  "PyYAML>=6.0"
  "numpy>=1.26.4,<2.0.0"
  "pydantic>=2.7"
  "pydantic-core==2.33.2"
  "pydub>=0.25"
  "mido>=1.3.0"
  "tomli>=2.0"
  "pretty_midi>=0.2.10"

  # music21 → matplotlib 連鎖依存
  "matplotlib>=3.8"
  "contourpy>=1.0.1"
  "fonttools>=4.22.0"
  "kiwisolver>=1.3.1"
  "Pillow>=10.0"

  # requests が内部で参照
  "charset_normalizer<4,>=2"
)

# ----------------------------------------------------------------------
#  前提チェック
# ----------------------------------------------------------------------
echo "チェック: ${WHEEL_DIR} が存在するか確認…"
if [[ ! -d "${PROJECT_ROOT}/${WHEEL_DIR}" ]]; then
  echo "ERROR: '${WHEEL_DIR}' ディレクトリがありません。" >&2
  exit 1
fi

# ----------------------------------------------------------------------
# 0. heavy パッケージを wheelhouse に揃える
# ----------------------------------------------------------------------
echo "0) heavy パッケージのホイールを準備中…"
for spec in "${HEAVY_PACKAGES[@]}"; do
  pkg="${spec%%[*<>=]*}"     # パッケージ名だけ
  pat="${PROJECT_ROOT}/${WHEEL_DIR}/${pkg}"*-cp312*manylinux_2_17_x86_64*.whl

  # 既にホイールがあればスキップ
  if ls $pat &>/dev/null; then
    continue
  fi

  if [[ "$pkg" == "pretty_midi" ]]; then
    echo "   → $pkg: wheel をローカルビルド"
    python -m pip wheel \
      --wheel-dir "${WHEEL_DIR}" \
      --no-deps "$spec"
  else
    echo "   → $pkg: manylinux ホイールをダウンロード"
    python -m pip download \
      --dest "${WHEEL_DIR}" \
      --platform manylinux_2_17_x86_64 \
      --implementation cp \
      --abi cp312 \
      --only-binary=:all: \
      --no-deps "$spec"
  fi
done

# ----------------------------------------------------------------------
# 1. pip / setuptools を wheelhouse で更新
# ----------------------------------------------------------------------
echo "1) pip / setuptools をアップグレード…"
python -m pip install \
  --no-index \
  --find-links="${WHEEL_DIR}" \
  --upgrade pip setuptools

# ----------------------------------------------------------------------
# 2. requirements.txt をオフラインインストール
# ----------------------------------------------------------------------
echo "2) 依存パッケージをインストール…"
python -m pip install \
  --no-index \
  --find-links="${WHEEL_DIR}" \
  -r "${REQ_FILE}"

# ----------------------------------------------------------------------
# 3. プロジェクト本体を editable で投入
# ----------------------------------------------------------------------
echo "3) プロジェクトを editable (-e) でインストール…"
python -m pip install --no-build-isolation --no-deps -e .
