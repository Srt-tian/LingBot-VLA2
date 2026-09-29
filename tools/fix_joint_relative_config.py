"""Upgrade only the five generated, not-yet-normalized Piper configs."""
from pathlib import Path
from prepare_piper_views import PROMPTS, robot_config

root = Path('/pfs/user/data/lingbot_vla_v2/piper_five_tasknorm')
updates = []
for task in PROMPTS:
    path = root / 'robot_configs' / f'piper_{task}.yaml'
    expected = robot_config(task, root)
    old = expected.replace('      relative_type: null\n', '')
    assert not (root / 'norm' / f'{task}.json').exists(), 'Norm must be recomputed after semantic changes'
    assert not path.is_symlink() and path.read_text() == old, path
    updates.append((path, expected))
for path, expected in updates:
    backup = path.with_suffix('.before_joint_relative.yaml')
    assert not backup.exists(), backup
    backup.write_text(path.read_text())
    path.write_text(expected)
    print(path)
