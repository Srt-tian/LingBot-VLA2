# Piper five-task CLOUD 30k training

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

Confirmed preparation recipe: 8 A800 GPUs, microbatch2/global16, Muon5e-5,
30000steps, save10000, SDPA vision/Flex train, FSDP2, same teachers and losses.
No automatic resume from the bottles smoke: all five start from official base.
W&B online project lingbot-vla2; key/entity injected through EIP environment.
Entrypoint: bash tools/run_piper_cloud30k.sh; TASK/OUTPUT_DIR required.
Each task gets a distinct output; entrypoint refuses existing output directories.

Before submission: verify assets, exact source commit recoverable from origin,
registry digest, cloud mount visibility, queue capacity and full resolved resources.
Display resolved configuration and obtain confirmation per EIP skill.
Freeze this checkout while any submitted task reads it. No edit/pull/checkout.
Never change existing MagicVLA repos, inference servers or robot processes.

Migration staging: /pfs/user/data/lingbot_vla_v2/migration_20261007.
TOS prefix: tos://magiclab/shenrongtian/lingbot_vla2_cloud_20261007/.
Only direct tosutil cp is used for bulk data; workstation carries small code/metadata.
Transfer completion and canonical publication must be independently verified.
