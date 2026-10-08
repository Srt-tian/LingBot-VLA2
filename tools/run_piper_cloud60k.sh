#!/usr/bin/env bash
set -Eeuo pipefail
PHASE=bootstrap
# Never print BASH_COMMAND or enable xtrace: commands can contain W&B secrets.
trap 'rc=$?; printf "STARTUP_FAILED phase=%s line=%s exit=%s\n" "$PHASE" "$LINENO" "$rc" >&2; exit "$rc"' ERR
printf 'BOOTSTRAP_START utc=%s host=%s\n' "$(date -u +%FT%TZ)" "$(hostname)"
TASK=${TASK:?Set TASK to bottles/pens/pepper/kitchen/lemon}
case "$TASK" in bottles|pens|pepper|kitchen|lemon) ;; *) exit 2 ;; esac
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:?Set unique OUTPUT_DIR}
case "$OUTPUT_DIR" in /workspace/user/experiments/lingbot_vla_v2/*) ;; *) exit 2 ;; esac
PHASE=bootstrap_logging
# Keep preflight errors even when the TOS output mount is unavailable.
mountpoint -q /pfs/user
BOOTSTRAP_DIR=/pfs/user/experiments/lingbot_vla_v2/bootstrap
mkdir -p "$BOOTSTRAP_DIR"
BOOTSTRAP_LOG=$(mktemp "$BOOTSTRAP_DIR/$(basename "$OUTPUT_DIR").XXXXXX.log")
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1
printf 'BOOTSTRAP_LOG=%s\n' "$BOOTSTRAP_LOG"
PHASE=workspace_mount
# Worker TOS implementations need not use the development host's FSTYPE.
findmnt -n -o TARGET,SOURCE,FSTYPE -T /workspace/user
mountpoint -q /workspace/user
test -w /workspace/user
PHASE=output_and_credentials
test ! -e "$OUTPUT_DIR"
: "${WANDB_API_KEY:?Inject W&B key through EIP environment}"
: "${WANDB_ENTITY:?Set W&B entity}"
export WANDB_PROJECT=lingbot-vla2 WANDB_MODE=online
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO_DIR"
PHASE=git_check
test -z "$(git status --porcelain)"
git log -1 --format='EXECUTION_COMMIT=%H'
PHASE=gpu_and_assets
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
PHASE=create_output
mkdir -p "$OUTPUT_DIR"
# Frequent small writes must not use the TOS-backed checkpoint mount.
RUNTIME_LOG_DIR=/pfs/user/experiments/lingbot_vla_v2/runtime_logs/$(basename "$OUTPUT_DIR")
mkdir -p "$RUNTIME_LOG_DIR/tensorboard" "$RUNTIME_LOG_DIR/wandb"
export LINGBOT_TB_DIR="$RUNTIME_LOG_DIR/tensorboard"
export WANDB_DIR="$RUNTIME_LOG_DIR/wandb"
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000 TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$RUNTIME_LOG_DIR/nccl_trace_"
export PYTHONFAULTHANDLER=1
printf 'RUNTIME_LOG_DIR=%s\n' "$RUNTIME_LOG_DIR"
PHASE=training
torchrun --standalone --nnodes=1 --nproc-per-node=8 tasks/vla/train_lingbotvla.py \
  "configs/vla/real_robot/piper_${TASK}_cloud60k.yaml" \
  --train.output_dir "$OUTPUT_DIR" \
  --train.wandb_name "$(basename "$OUTPUT_DIR")" 2>&1 | tee "$RUNTIME_LOG_DIR/train.log"
