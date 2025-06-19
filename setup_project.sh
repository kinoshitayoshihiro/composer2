#!/usr/bin/env bash
set -euo pipefail

WHEEL_DIR="wheelhouse"     # Codespace 内ではリポジトリ直下
VENV_DIR=".venv"
REQ_FILE="requirements.txt"
OUT_DIR="midi_output"

# 1. venv 作成
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# 2. wheelhouse を整理（再度呼んでも冪等なので OK）
python scripts/clean_wheels.py

# 3. pip / setuptools 更新
pip install --no-index --find-links="${WHEEL_DIR}" --upgrade pip setuptools

# 4. 依存インストール
pip install --no-index --find-links="${WHEEL_DIR}" -r "${REQ_FILE}"

# 5. プロジェクト (-e)
pip install --no-build-isolation --no-deps -e .

mkdir -p "${OUT_DIR}"
echo "✅ Setup finished.  Run:  source ${VENV_DIR}/bin/activate"
