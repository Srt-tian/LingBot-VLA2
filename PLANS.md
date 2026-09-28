# LingBot-VLA 2.0 Piper five-task IDC smoke

This checkout belongs to this training preparation only. Do not touch existing
MagicVLA training checkouts, inference services, containers, datasets, or norms.

## Source and runtime

- Upstream: https://github.com/Robbyant/lingbot-vla-v2.git
- Starting commit: be969b8fd117fb70550c5d4bf4bc328211b5b1b6
- Branch: train/piper-five-idc-smoke
- User-authorized writable remote: https://github.com/Srt-tian/LingBot-VLA2.git
- User now requests local IDC commits during iteration, with one consolidated
  GitHub push after stabilization. Do not push each preparation change.
- Runtime target: Python 3.12, Torch 2.8.0+cu126, FlashAttention 2.8.3.
- Upstream docker/Dockerfile is stale (Torch 2.5.1); use the independent
  docker/Dockerfile.idc after verifying the wheelhouse.
- Download/build staging: /pfs/user/data/lingbot_vla_v2/build.
- Model roots: /pfs/user/data/lingbot_vla_v2/models.
- Preserve requirements.txt's huggingface-hub 0.34.3 rather than silently applying
  create_train_env.sh's conflicting final downgrade to 0.34.0; validate imports
  and the actual training path. Image is unvalidated until these tests pass.
- Image build must not assume GPU availability. GPU checks run afterward.
- IDC dataset image uses LeRobot 0.3.3 (v2.1). Official script's 0.4.2 rejects
  v2.1. The compatibility patch derives shared-video offsets from dataset
  codebase_version rather than import namespace, and supports 0.3.3 constants.
  Full loader verification is still required; do not claim runtime acceptance.

## Data and acceptance

Five separate task models: bottles, pens, pepper, kitchen, lemon. Use each task's
full source collection with one shared task norm. Do not reuse MagicVLA's 34-D
mapping or normalization: LingBot-VLA 2.0 has a 55-D canonical representation.
Verify raw joint/gripper meanings, padding masks, camera names, prompt, frame
timing, and relative arm/absolute gripper semantics against official code.

Pending: dataset audit, mapping tests, norms, base/Qwen/MoGe/depth/video assets,
image build/runtime validation, loader checks, and exact smoke configuration.
Smoke must exercise finite forward/backward/optimizer updates and checkpoint
save/reload; five formal CLOUD tasks follow only after acceptance.

No EIP task is authorized by the existence of this plan. Present full resolved
IDC resources and inputs, obtain confirmation, and record the task in the
sanitized local EIP ledger. Publish execution commits to an authorized canonical
remote before training submission; the upstream official repository is read-only.

## Completed preparation checks

- All 1676 episodes / 973824 frames passed shape, finite-value, episode-index,
  timestamp monotonicity, and metadata-count checks (CPU-only parquet audit).
- Upstream pad_and_concat / prepare_state / prepare_action isolated unit test
  passed: state55, action50x55, valid slots0..11 and28,29, all inactive slots zero.
- Native VLA base/depth/DINO-video, Qwen and MoGe weights checksum verified.
  Qwen last shard was recovered by checked Content-Range resume. Never use a
  .partial artifact.
- First wheelprep stopped on pip read timeout. Preserve cache, use 120s timeout,
  no automated retries, fetch only Torch/vision/audio/Triton from cu126 index,
  and resolve remaining packages from PyPI to avoid unnecessary CDN transfers.
- Exclude MLflow from the VLA runtime: only standalone MoGe train.py uses it;
  full/skinny variants conflict with upstream PyArrow21/Packaging25 respectively.
- Pin plyfile1.1.2 (declares numpy>=1.21), not1.1.4 (requires numpy>=2), preserving
  the model environment's NumPy1.26.4. Verify depth/runtime imports after build.
