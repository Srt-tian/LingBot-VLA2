#!/usr/bin/env bash
# Run inside the already-present CUDA 12.6 devel container with /stage mounted
# from /pfs/user/data/lingbot_vla_v2/build. Never download packages locally.
set -Eeuo pipefail
export TMPDIR=/stage/tmp PIP_CACHE_DIR=/stage/pip-cache
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=0
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" /stage/wheels
if [[ ! -x /stage/bootstrap/bin/python ]]; then
  bash /stage/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh -b -p /stage/bootstrap
fi
export PATH=/stage/bootstrap/bin:/usr/local/cuda/bin:$PATH
python -m pip download --no-deps --dest /stage/wheels --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 triton==3.4.0
python -m pip wheel --wheel-dir /stage/wheels --find-links /stage/wheels \
  -r /src/requirements.txt -r /src/docker/requirements-depth.idc.txt \
  setuptools==75.8.0 wheel==0.45.1
python -m pip download --no-deps --dest /stage/wheels lerobot==0.3.3 numpydantic==1.9.0
# FlashAttention needs Torch installed to compile its wheel. Pin the same stack
# used in the final image; all package caches and temporary downloads are on PFS.
python -m pip install --no-index --find-links /stage/wheels \
  torch==2.8.0 packaging==25.0 ninja==1.11.1.4 wheel setuptools
export FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST='8.0;8.9;9.0'
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /stage/wheels flash-attn==2.8.3
sha256sum /stage/wheels/* > /stage/wheels.sha256
echo WHEELHOUSE_READY
