#!/usr/bin/env bash
# Single-node four-GPU smoke + explicit resume. Submit only after EIP review.
set -Eeuo pipefail
REPO_DIR=${REPO_DIR:-/pfs/user/code/lingbot-vla-v2-piper-five}
OUTPUT_DIR=${OUTPUT_DIR:?Provide a unique, reviewed smoke output directory}
case "$OUTPUT_DIR" in
  /pfs/user/experiments/lingbot_vla_v2/*) ;;
  *) echo 'Output must be under the LingBot-VLA2 experiment root' >&2; exit 1 ;;
esac
test ! -e "$OUTPUT_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
CONFIG=configs/vla/real_robot/piper_bottles_sdpa_smoke.yaml
test -s /pfs/user/data/lingbot_vla_v2/piper_five_tasknorm/norm/bottles.json
python -c "import importlib.util,torch; assert importlib.util.find_spec('flash_attn') is None; assert torch.cuda.device_count()==4; print('FOUR_GPU_NO_EXTERNAL_FA2_PREFLIGHT_OK')"
mkdir -p "$OUTPUT_DIR"
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  tasks/vla/train_lingbotvla.py "$CONFIG" \
  --train.output_dir "$OUTPUT_DIR" --train.enable_resume false \
  2>&1 | tee "$OUTPUT_DIR/smoke_initial.log"
test -d "$OUTPUT_DIR/checkpoints/global_step_10"
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  tasks/vla/train_lingbotvla.py "$CONFIG" \
  --train.output_dir "$OUTPUT_DIR" --train.max_steps 12 \
  --train.load_checkpoint_path "$OUTPUT_DIR/checkpoints/global_step_10" \
  2>&1 | tee "$OUTPUT_DIR/smoke_resume.log"
test -d "$OUTPUT_DIR/checkpoints/global_step_12"
echo SMOKE_TRAIN_AND_RESUME_FINISHED_REVIEW_METRICS
