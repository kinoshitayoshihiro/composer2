#!/usr/bin/env bash
set -eux
python3 -m pip install --break-system-packages --upgrade pip --ignore-installed
# CPU-only PyTorch first to avoid CUDA deps
python3 -m pip install --break-system-packages --index-url https://download.pytorch.org/whl/cpu "torch==2.3.*"
python3 -m pip install --break-system-packages "numpy<2.3" numba==0.59.1
python3 -m pip install --break-system-packages -r requirements.txt
