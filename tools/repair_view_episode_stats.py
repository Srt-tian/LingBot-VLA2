"""Recompute consumed numeric metadata in isolated views, never raw datasets.

This is LeRobot metadata compatibility, NOT the training action-chunk norm.
The complete original statistics are preserved beside each corrected file.
"""
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/pfs/user/data/lingbot_vla_v2/piper_five_tasknorm')
manifest = json.loads((ROOT / 'manifest.json').read_text())
keys = ['observation.qpos', 'action']
for spec in manifest.values():
    for view_name in spec['views']:
        view = Path(view_name)
        assert view.resolve().is_relative_to(ROOT / 'views')
        assert not (view / 'meta').is_symlink()
        original = view / 'meta/episodes_stats.jsonl'
        backup = original.with_suffix('.before_lerobot_compat.jsonl')
        assert original.is_file() and not original.is_symlink()
        assert not backup.exists(), f'Already repaired or interrupted: {backup}'
for task, spec in manifest.items():
    for view_name in spec['views']:
        view = Path(view_name)
        info = json.loads((view / 'meta/info.json').read_text())
        episodes = [json.loads(line) for line in (view / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        records = []
        frames = 0
        for episode in episodes:
            idx = episode['episode_index']
            relative = info['data_path'].format(episode_chunk=idx // info.get('chunks_size', 1000), episode_index=idx)
            data = pq.read_table(view / relative, columns=keys)
            assert len(data) == episode['length']
            stats = {}
            for key in keys:
                array = np.asarray(data[key].to_pylist(), dtype=np.float64)
                assert array.shape == (len(data), 14) and np.isfinite(array).all()
                stats[key] = {'min': array.min(0).tolist(), 'max': array.max(0).tolist(),
                              'mean': array.mean(0).tolist(), 'std': array.std(0).tolist(),
                              'count': [len(array)]}
            records.append({'episode_index': idx, 'stats': stats})
            frames += len(data)
        assert frames == info['total_frames']
        original = view / 'meta/episodes_stats.jsonl'
        shutil.copy2(original, original.with_suffix('.before_lerobot_compat.jsonl'))
        temporary = original.with_suffix('.compat.tmp')
        with temporary.open('x') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
        temporary.replace(original)
        print(json.dumps({'task': task, 'view': view_name, 'episodes': len(records),
                          'frames': frames, 'status': 'metadata_recomputed'}), flush=True)
print('VIEW_METADATA_RECOMPUTE_PASSED; TRAINING_NORMS_STILL_SEPARATE', flush=True)
