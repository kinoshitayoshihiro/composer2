# --- ベースイメージ ---
FROM mcr.microsoft.com/devcontainers/python:3.10

# 作業ディレクトリ
WORKDIR /workspace/composer2

# ↑ ここで appuser はあらかじめ DevContainer が用意済み
#    （通常、devcontainers イメージは非 root ユーザー 'vscode' を使うので、
#     必要なら USER vscode にしておいてください）

# オフラインバンドルをコピー
COPY wheelhouse ./wheelhouse

# ソースをコピー
COPY . .

# ユーザーのホーム配下に venv を作り、オフラインインストール
USER vscode
RUN python -m venv .venv \
 && . .venv/bin/activate \
 && pip install --upgrade pip setuptools \
 && pip install --no-index --find-links=./wheelhouse -r requirements.txt

# 仮想環境の bin を PATH に追加
ENV PATH=/workspace/composer2/.venv/bin:$PATH

# （オプション）出力フォルダを作成
RUN mkdir -p midi_output
