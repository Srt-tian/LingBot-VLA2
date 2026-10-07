#!/usr/bin/env bash
set -Eeuo pipefail
TASK=${TASK:?Set TASK to bottles/pens/pepper/kitchen/lemon}
case "$TASK" in bottles|pens|pepper|kitchen|lemon) ;; *) exit 2 ;; esac
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:?Set unique OUTPUT_DIR}
case "$OUTPUT_DIR" in /workspace/user/experiments/lingbot_vla_v2/*) ;; *) exit 2 ;; esac
# Do not silently write checkpoints to a worker's container root filesystem.
test "$(findmnt -n -o TARGET -T /workspace/user)" = /workspace/user
test "$(findmnt -n -o FSTYPE -T /workspace/user)" = hpvs_fs
test -w /workspace/user
findmnt -n -o TARGET,SOURCE,FSTYPE -T /workspace/user
test ! -e "$OUTPUT_DIR"
: "${WANDB_API_KEY:?Inject W&B key through EIP environment}"
: "${WANDB_ENTITY:?Set W&B entity}"
export WANDB_PROJECT=lingbot-vla2 WANDB_MODE=online
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO_DIR"
test -z "$(git status --porcelain)"
git log -1 --format='EXECUTION_COMMIT=%H'
python - "$TASK" <<'PY'
import importlib.util,json,sys
from pathlib import Path
import torch,yaml
task=sys.argv[1]
assert torch.cuda.device_count()==8
assert importlib.util.find_spec('flash_attn') is None
root=Path('/pfs/user/data/lingbot_vla_v2')
norm=root/'piper_five_tasknorm/norm'/f'{task}.json'
assert json.loads(norm.read_text())
for line in (root/'piper_five_tasknorm/lists'/f'{task}.txt').read_text().splitlines():
    name,path=line.split(); p=Path(path)
    for part in ['data','videos','meta/info.json']: assert (p/part).exists(),str(p/part)
    cfg=yaml.safe_load((root/'piper_five_tasknorm/robot_configs'/f'{name}.yaml').read_text())
    assert cfg['norm_stats']==str(norm)
for path in ['lingbot-vla-v2-6b','Qwen3-VL-4B-Instruct','moge-2-vitb-normal/model.pt']:
    assert (root/'models'/path).exists(),path
print('EIGHT_GPU_TASKNORM_PREFLIGHT_OK',task,flush=True)
PY
mkdir -p "$OUTPUT_DIR"
export WANDB_DIR="$OUTPUT_DIR"
torchrun --standalone --nnodes=1 --nproc-per-node=8 tasks/vla/train_lingbotvla.py \
  "configs/vla/real_robot/piper_${TASK}_cloud30k.yaml" \
  --train.output_dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/train.log"
