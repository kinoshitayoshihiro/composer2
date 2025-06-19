# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.12

# wheelhouse をコピー
COPY wheelhouse /opt/wheelhouse
COPY scripts/clean_wheels.py /opt/clean_wheels.py   # ← 追加

# ネイティブライブラリ (scipy などが参照) を apt で
RUN apt-get update && \
      apt-get install -y --no-install-recommends \
      libsndfile1 libjpeg62-turbo && \
      apt-get clean && rm -rf /var/lib/apt/lists/*

# wheel 整理（イメージを小さくするため）
RUN python /opt/clean_wheels.py

WORKDIR /workspace
