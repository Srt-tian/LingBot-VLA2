"""Read-only audit of the 14 existing IDC LeRobot roots; never edits data."""
import json
from pathlib import Path

RAW = Path('/pfs/user/data/host_piper_paper/raw')
EXTRA = Path('/pfs/user/data/magicvla_dynamic_v2_5tasks_20260917')
SOURCES = {
    'bottles': [RAW / x for x in ['bottles_task_1819_teleop_20260806', 'bottles_task_1824_teleop_20260806-2', 'bottles_task_1831_teleop_20260806', 'bottles_task_1842_teleop_20260807']],
    'pens': [RAW / 'pens_task_1841_teleop_20260807'],
    'pepper': [Path('/pfs/user/data/shenrongtian_20260914162718_CTP/lerobot_merge_result')],
    'kitchen': [EXTRA / 'kitchen' / x for x in ['task_2099/teleop_20260827', 'task_2112/teleop_20260828', 'task_2113/teleop_20260828', 'task_2120/teleop_20260829', 'task_2121/teleop_20260829', 'task_2122/teleop_20260830']],
    'lemon': [EXTRA / 'lemon/songling/task_1700/teleop_20260804', EXTRA / 'lemon/teleop/task_1895/teleop_20260813'],
}

def main():
    for task, roots in SOURCES.items():
        episodes = frames = 0
        for root in roots:
            meta = json.loads((root / 'meta/info.json').read_text())
            prompts = [json.loads(line) for line in (root / 'meta/tasks.jsonl').read_text().splitlines() if line.strip()]
            features = meta['features']
            episodes += meta['total_episodes']
            frames += meta['total_frames']
            print(json.dumps({'task': task, 'root': str(root), 'version': meta['codebase_version'], 'episodes': meta['total_episodes'], 'frames': meta['total_frames'], 'fps': meta['fps'], 'features': {k: v.get('shape') for k,v in features.items() if k in ['observation.state','observation.qpos','action','real_action','state_end','action_end'] or k.startswith('observation.images.')}, 'tasks': prompts}, ensure_ascii=False), flush=True)
        print(json.dumps({'summary': task, 'roots': len(roots), 'episodes': episodes, 'frames': frames}), flush=True)


if __name__ == '__main__':
    main()
