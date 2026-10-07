# Piper five-task CLOUD official real-robot training

Authoritative checkout: /pfs/user/code/lingbot-vla-v2-piper-five-cloud.
Branch: train/piper-five-cloud30k. Origin: https://github.com/Srt-tian/LingBot-VLA2.git.
Base implementation: 79be3a97aeafedb2da3f2335c4082be6c89af037.
IDC task13555 passed four-GPU train10/save10/reload10/train12/save12; runtime9m12s,
reported peak37.64GB, finite loss/GradNorm. This does not prove eight-GPU training.

Five separate base-initialized models: bottles/pens/pepper/kitchen/lemon.
Per task: full source collection, one shared task-specific norm; never mix norms.
Use the migrated LingBot55 joint representation, not MagicVLA34.
Metadata/index alignment adapter and source joint delta/gripper absolute semantics
are unchanged from successful smoke. Original payloads stay read-only.

User superseded the smoke-derived recipe with official real-robot settings.
Reference: Robbyant/lingbot-vla-v2 configs/vla/real_robot/real_robot.yaml,
upstream base be969b8fd117fb70550c5d4bf4bc328211b5b1b6, also verified on main
2026-10-07. Training_Config.md embeds a differing example; actual YAML is authoritative.
Recipe: 8 A800 GPUs, microbatch16/gradient_accumulation_steps2/global256,
Muon5e-5, constant LR,
60000steps, save20000, num_workers8, enable_resume=true,
gradient checkpointing=false as in official YAML; SDPA vision/Flex train,
FSDP2, same teachers and losses. Task13559 microbatch32 failed at the first
forward with CUDA OOM (79.29GiB process use on79.33GiB device); no completed step.
User requested microbatch16; accumulation2 preserves official global256.
This memory adaptation is configured for all five tasks, not yet runtime tested.
If memory adaptation is needed, propose it explicitly rather than silently
reducing global batch or training duration. Existing 30k configs are historical.
No automatic resume from the bottles smoke: all five start from official base.
W&B online project lingbot-vla2; key/entity injected through EIP environment.
Entrypoint: bash tools/run_piper_cloud60k.sh; TASK/OUTPUT_DIR required.
Configs: configs/vla/real_robot/piper_<task>_cloud60k.yaml.
Despite official enable_resume=true, a new unique output is mandatory, so fresh
runs initialize official base, not old checkpoints.
Each task gets a distinct output; entrypoint refuses existing output directories.
Output override: /workspace/user/experiments/lingbot_vla_v2/<unique-run>.
The CLOUD development machine has an hpvs_fs mount at /workspace/user,
source BFD01CD151:magiclab:/shenrongtian. Never use the /workspace root.
Entrypoint requires an actual writable mount at /workspace/user on the
training worker too, but does not pin the development host's hpvs_fs FSTYPE.
Bootstrap logging starts before workspace preflight and is persisted under
/pfs/user/experiments/lingbot_vla_v2/bootstrap. An ERR trap reports phase,
line and exit code, never shell command expansion or credential values.
Task13557/13558 failed in about16s with no application logs; the exact original
root cause is not proven. The FSTYPE check is removed as an unjustified constraint,
not described as a verified fix for those failures. Official hyperparameters unchanged.
YAML output_dir is overridden by required OUTPUT_DIR at launch. Checkpoint
save/reload on this storage has not yet been validated; IDC smoke used PFS.

Before submission: verify assets, exact source commit recoverable from origin,
registry digest, cloud mount visibility, queue capacity and full resolved resources.
Display resolved configuration and obtain confirmation per EIP skill.
Freeze this checkout while any submitted task reads it. No edit/pull/checkout.
Never change existing MagicVLA repos, inference servers or robot processes.

Migration staging: /pfs/user/data/lingbot_vla_v2/migration_20261007.
TOS prefix: tos://magiclab/shenrongtian/lingbot_vla2_cloud_20261007/.
Only direct tosutil cp is used for bulk data; workstation carries small code/metadata.
Transfer completion and canonical publication must be independently verified.
