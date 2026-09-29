# Build with named context "deps" pointing at the verified PFS build directory.
# docker build --build-context deps=/pfs/user/data/lingbot_vla_v2/build \
#   -f docker/Dockerfile.idc -t <registry-tag> .
# Packages are downloaded by tools/prepare_wheels.sh on IDC, never by this build.
FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    PYTHONNOUSERSITE=1 PIP_NO_INPUT=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
COPY --from=deps Miniconda3-py312_25.1.1-2-Linux-x86_64.sh /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
WORKDIR /opt/lingbot-vla-v2
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 libgomp1 libexpat1 build-essential git pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/lingbot-requirements.txt
COPY docker/requirements-depth.idc.txt /tmp/lingbot-depth-requirements.txt
RUN --mount=type=bind,from=deps,source=wheels,target=/wheels \
    python -m pip install --no-index --find-links=/wheels \
      -r /tmp/lingbot-requirements.txt -r /tmp/lingbot-depth-requirements.txt setuptools==75.8.0 wheel==0.45.1 \
    && python -m pip install --no-index --find-links=/wheels --no-deps \
      lerobot==0.3.3 numpydantic==1.9.0
COPY . .
RUN python -m pip install --no-build-isolation --no-deps -e . \
      -e lingbotvla/models/vla/vision_models/lingbot-depth \
      -e lingbotvla/models/vla/vision_models/MoGe
RUN python -c "import site,pathlib; pathlib.Path(site.getsitepackages()[0], 'stablevla_local_depth.pth').write_text('/opt/lingbot-vla-v2/lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d\\n')"
RUN python -c "import importlib.util; assert importlib.util.find_spec('flash_attn') is None; import torch,transformers,cv2,accelerate,trimesh,moge,mdm,utils3d; assert torch.__version__.split('+')[0]=='2.8.0'; print(torch.__version__,transformers.__version__,'NO_EXTERNAL_FA2')"
CMD ["/bin/bash"]
