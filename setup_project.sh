#!/usr/bin/env bash
# =====================================================================
#  setup_project.sh  ─ Robust bootstrap (offline / online compatible)
# ---------------------------------------------------------------------
#  • Python venv を安全に作成（bin/python が無い環境も吸収）
#  • 重い依存を wheelhouse にキャッシュし、オフラインでも再利用
#  • requirements.txt＋編集モードでプロジェクトをインストール
# =====================================================================
set -euo pipefail

# ---------- 0. Paths --------------------------------------------------
ROOT_DIR="$(pwd)"
WHEEL_DIR="$ROOT_DIR/wheelhouse"
REQ_FILE="$ROOT_DIR/requirements.txt"
OUTPUT_DIR="$ROOT_DIR/midi_output"

VENV_DIR="$ROOT_DIR/.venv"
VPY="$VENV_DIR/bin/python"
VPIP="$VENV_DIR/bin/pip"

PYTAG="cp311"                 # 変える場合はここ
MANYLINUX="manylinux2014_x86_64"

# ---------- 1. Heavy wheels -------------------------------------------
HEAVY_PKGS=(
  # 基本
  "wheel>=0.43" "pip>=24.0" "setuptools>=68.0"
  # 科学計算
  "numpy>=1.26.4,<2.0.0" "scipy>=1.10"
  # コア
  "PyYAML>=6.0" "tomli>=2.0"
  "pydantic>=2.7" "pydantic-core==2.33.2"
  # オーディオ
  "pretty_midi>=0.2.10" "mido>=1.3.0" "pydub>=0.25"
  "soundfile>=0.12" "audioread>=2.1.9"
  # librosa & 依存
  "numba>=0.57" "llvmlite>=0.44" "librosa>=0.10"
  # プロット
  "matplotlib>=3.8" "contourpy>=1.3.0"
  "fonttools>=4.58.0" "kiwisolver>=1.4" "Pillow>=11.0"
  # ネットワーク
  "charset_normalizer>=2,<4"
)

# ---------- 2. Create/Fix venv ----------------------------------------
if [[ ! -x "$VPY" ]]; then
  echo "🟢 creating venv ($VENV_DIR)"

  # ❶ まず標準（コピー）モード
  if ! python3 -m venv "$VENV_DIR"; then
    echo "⚠️  copy-mode failed, retrying with --symlinks"
    python3 -m venv --symlinks "$VENV_DIR"
  fi

  # ❷ bin/python or bin/python3 が欠けていればリンクで補完
  [[ -x "$VENV_DIR/bin/python"  ]] || ln -s python3 "$VENV_DIR/bin/python"  || true
  [[ -x "$VENV_DIR/bin/python3" ]] || ln -s python  "$VENV_DIR/bin/python3" || true
fi

echo "   venv python: $("$VPY" -V)"

# ---------- 3. ensure pip/wheel/setuptools ----------------------------
"$VPY" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VPY" -m pip install --upgrade --quiet wheel pip setuptools

# ---------- 4. ONLINE / OFFLINE 判定 ---------------------------------
if [[ ! -d "$WHEEL_DIR" || -z "$(ls -A "$WHEEL_DIR" 2>/dev/null)" ]]; then
  OFFLINE=false
  mkdir -p "$WHEEL_DIR"
  echo "🟡 wheelhouse empty → ONLINE mode"
else
  OFFLINE=true
  echo "🟢 wheelhouse found → OFFLINE mode"
fi

# ---------- 5. ダウンロード（オンラインのみ） ----------------------
if ! $OFFLINE; then
  echo "🟢 downloading heavy wheels"
  for spec in "${HEAVY_PKGS[@]}"; do
    pkg=${spec%%[<>=]*}
    if compgen -G "$WHEEL_DIR/${pkg}-*-${PYTAG}-*manylinux*.whl" >/dev/null; then
      continue  # 既にキャッシュあり
    fi
    echo "   → $pkg"
    "$VPY" -m pip download --dest "$WHEEL_DIR"                 \
           --platform "$MANYLINUX" --implementation cp         \
           --abi "$PYTAG" --python-version 3.11                \
           --only-binary=:all: --no-deps "$spec" || true
  done
fi

# ---------- 6. venv へ wheelhouse で基本ツール更新 -------------------
"$VPIP" install --no-index --find-links "$WHEEL_DIR"           \
       --upgrade --quiet wheel pip setuptools

# ---------- 7. requirements.txt をインストール ------------------------
echo "🟢 installing project requirements"
if $OFFLINE; then
  "$VPIP" install --no-index --find-links "$WHEEL_DIR" -r "$REQ_FILE"
else
  "$VPIP" install -r "$REQ_FILE"
fi

# ---------- 8. プロジェクトを editable で入れる -----------------------
"$VPIP" install --no-build-isolation --no-deps -e "$ROOT_DIR"

# ---------- 9. 生成フォルダなど --------------------------------------
mkdir -p "$OUTPUT_DIR"

echo "✅ setup finished – run  'source $VENV_DIR/bin/activate'"
