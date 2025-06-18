FROM mcr.microsoft.com/devcontainers/python:3.10

# Dockerfile の適当な場所に
COPY wheelhouse /workspace/composer2/wheelhouse

WORKDIR /workspace/composer2


# wheelhouse も同時にイメージ内に取り込む
COPY wheelhouse ./wheelhouse

# 必要なら追加のパッケージをここでインストール