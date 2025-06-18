FROM mcr.microsoft.com/devcontainers/python:3.10

# 必要なら追加のパッケージをここでインストール

WORKDIR /workspace/composer2

# Dockerfile の適当な場所に
COPY wheelhouse /workspace/composer2/wheelhouse


# wheelhouse も同時にイメージ内に取り込む
COPY wheelhouse ./wheelhouse

# 以降のセットアップは postCreateCommand で
