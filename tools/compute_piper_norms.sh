#!/usr/bin/env bash
# Execute inside the validated LingBot-VLA2 image, with IDC PFS mounted.
# Staged helper: not accepted until actual loader/runtime checks pass.
set -Eeuo pipefail
REPO_DIR=${REPO_DIR:-/opt/lingbot-vla-v2}
DATA_ROOT=/pfs/user/data/lingbot_vla_v2/piper_five_tasknorm
WORK_ROOT=/pfs/user/data/lingbot_vla_v2/preparation/norm_compute
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
test -f "$REPO_DIR/scripts/compute_norm_stats.py"
mkdir -p "$WORK_ROOT" "$DATA_ROOT/norm"
cd "$WORK_ROOT"
for task in bottles pens pepper kitchen lemon; do
  output="$DATA_ROOT/norm/$task.json"
  if [[ -e "$output" ]]; then
    echo "Refusing to replace existing norm: $output" >&2
    exit 1
  fi
  test -s "$DATA_ROOT/lists/$task.txt"
  test -s "$DATA_ROOT/robot_configs/piper_$task.yaml"
done
for task in bottles pens pepper kitchen lemon; do
  python "$REPO_DIR/scripts/compute_norm_stats.py" \
    "$REPO_DIR/configs/vla/norm_compute/post_data.yaml" \
    --data.robot_name "piper_$task" \
    --data.train_path "$DATA_ROOT/lists/$task.txt" \
    --data.robot_config_root "$DATA_ROOT/robot_configs" \
    --data.norm_path "$DATA_ROOT/norm/$task.json" \
    --data.num_workers 2 \
    --data.data_ratio_for_norm_compute 1.0 \
    --data.norm_merge_chunk_dim true \
    --train.chunk_size 50 \
    --train.micro_batch_size 48 \
    2>&1 | tee "$WORK_ROOT/$task.log"
  test -s "$DATA_ROOT/norm/$task.json"
done
echo FIVE_TASK_NORMS_WRITTEN
