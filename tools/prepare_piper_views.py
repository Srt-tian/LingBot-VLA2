"""Create isolated task-labelled LeRobot metadata views; payloads stay read-only.

No source metadata or payload is changed. Every task has one shared norm path.
Configs follow upstream's compact arm-feature packing, NOT MagicVLA's layout.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
from audit_piper_sources import SOURCES

PROMPTS = {
    'bottles': 'Stand the bottles upright on the table.',
    'pens': 'Put the pens into the pen holder.',
    'pepper': 'Pick up the green pepper.',
    'kitchen': 'Place the seasoning containers on the storage rack.',
    'lemon': 'Put the lemon onto the plate.',
}


def robot_config(task, prefix):
    pieces = []
    for category, target, source in [('states', 'observation.state', 'observation.qpos'), ('actions', 'action', 'action')]:
        pieces.append(category + ':')
        for feature, slices in [('arm.position', [(0, 6), (7, 13)]), ('effector.position', [(6, 7), (13, 14)])]:
            pieces.extend([f'  - {target}.{feature}:', '      origin_keys:'])
            for start, end in slices:
                pieces.extend([f'        - {source}:', f'            start: {start}', f'            end: {end}'])
            if category == 'actions':
                pieces.append('      subtract_state: ' + ('true' if feature == 'arm.position' else 'false'))
    raw_cameras = ['cam_high', 'cam_left_wrist', 'cam_right_wrist'] if task == 'pepper' else ['cam_front', 'cam_left', 'cam_right']
    pieces.append('images:')
    for target, raw in zip(['camera_top', 'camera_wrist_left', 'camera_wrist_right'], raw_cameras):
        pieces.extend([f'  - observation.images.{target}:', f'      origin_keys: observation.images.{raw}'])
    pieces.append(f'norm_stats: {prefix}/norm/{task}.json')
    return '\n'.join(pieces) + '\n'


def prepare(root):
    if socket.gethostname() != 'dev-instance-shenrongtian' or not root.resolve().is_relative_to(Path('/pfs/user/data')):
        raise RuntimeError('Run on designated IDC host under /pfs/user/data only')
    if root.exists():
        raise RuntimeError('Destination already exists; refusing to overwrite')
    # Validate all roots before creating anything.
    for roots in SOURCES.values():
        for source in roots:
            meta = json.loads((source / 'meta/info.json').read_text())
            assert meta['features']['observation.qpos']['shape'] == [14]
            assert meta['features']['action']['shape'] == [14]
            assert meta['codebase_version'] == 'v2.1'
            assert (source / 'meta/tasks.jsonl').is_file()
            assert (source / 'data').is_dir() and (source / 'videos').is_dir()
    root.mkdir(parents=True)
    for directory in ['robot_configs', 'lists', 'norm']:
        (root / directory).mkdir()
    report = {}
    for task, sources in SOURCES.items():
        views = []
        for index, source in enumerate(sources):
            view = root / 'views' / task / f'source_{index:02d}'
            view.mkdir(parents=True)
            shutil.copytree(source / 'meta', view / 'meta')
            for name in ['data', 'videos']:
                (view / name).symlink_to(source / name, target_is_directory=True)
            tasks_file = view / 'meta/tasks.jsonl'
            tasks = [json.loads(s) for s in tasks_file.read_text().splitlines() if s.strip()]
            for row in tasks:
                row['task'] = PROMPTS[task]
            tasks_file.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in tasks))
            episodes_file = view / 'meta/episodes.jsonl'
            if episodes_file.exists():
                episodes = [json.loads(s) for s in episodes_file.read_text().splitlines() if s.strip()]
                for episode in episodes:
                    if 'tasks' in episode:
                        episode['tasks'] = [PROMPTS[task]]
                episodes_file.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in episodes))
            views.append(str(view))
        (root / 'lists' / f'{task}.txt').write_text(''.join(f'piper_{task} {view}\n' for view in views))
        (root / 'robot_configs' / f'piper_{task}.yaml').write_text(robot_config(task, root))
        report[task] = {'prompt': PROMPTS[task], 'raw_sources': list(map(str, sources)), 'views': views, 'shared_norm': str(root / 'norm' / f'{task}.json'), 'norm_computed': False}
    (root / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root)
