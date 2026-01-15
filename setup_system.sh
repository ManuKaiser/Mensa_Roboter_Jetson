#!/usr/bin/env bash
set -e

############################
# 1. System dependencies
############################
sudo apt-get update
sudo apt-get install -y \
  python3-pip \
  libopenblas-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libwebp-dev \
  zlib1g-dev \
  libpython3-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  python3-opencv \
  ninja-build \
  git

############################
# 2. Python virtual environment
############################
VENV_PATH="social_lstm/.venv"

if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv "$VENV_PATH" --system-site-packages
fi

source "$VENV_PATH/bin/activate"

pip install --upgrade pip setuptools wheel

############################
# 3. PyTorch (NVIDIA wheel)
############################
pip install --no-cache-dir \
  https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

############################
# 4. torchvision from source
############################
if [ ! -d "torchvision" ]; then
  git clone --branch release/0.20 https://github.com/pytorch/vision torchvision
fi

pushd torchvision
export BUILD_VERSION=0.20.0
python setup.py install
popd

############################
# 5. Python packages (no dependency overrides)
############################
pip install --no-deps \
  numpy==1.24.4 \
  pandas \
  ultralytics \
  deep_sort_realtime \
  pyrealsense2

############################
# 6. Verification
############################
python - <<'EOF'
import cv2
import numpy as np
import torch
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN available:", torch.backends.cudnn.is_available())
EOF
