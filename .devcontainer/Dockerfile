# --- ビルドステージ ---
FROM python:3.12-slim AS builder

WORKDIR /app

# wheelhouse フォルダだけ先に作成＆コピー
RUN mkdir wheelhouse

# SciPy の manylinux ホイールを wheelhouse にダウンロード
# ※ビルド時はネットワーク ON の前提
RUN pip install --upgrade pip setuptools \
 && pip download --dest wheelhouse \
      --platform manylinux_2_17_x86_64 \
      --only-binary=:all: \
      --no-deps \
      'scipy>=1.10'

# requirements.txt ほかソース一式をコピー
COPY requirements.txt .
COPY . .

# 他の依存パッケージも同様にダウンロード
RUN pip download --dest wheelhouse \
      --platform manylinux_2_17_x86_64 \
      --only-binary=:all: \
      --no-deps \
      -r requirements.txt

# wheelhouse とホイールを使って依存をインストール
RUN pip install --no-index \
      --find-links=wheelhouse \
      -r requirements.txt

# --- ランタイムステージ ---
FROM python:3.12-slim

WORKDIR /app

# ビルドステージから site-packages をコピー
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
# ソースをコピー
COPY --from=builder /app /app

# 非 root ユーザー切り替えなど
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

CMD ["python", "modular_composer.py", "--help"]
