"""CPU-only numeric audit of every episode, without decoding videos."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from audit_piper_sources import SOURCES

report = {}
for task, roots in SOURCES.items():
    entry = {'episodes': 0, 'frames': 0, 'roots': []}
    extrema = {k: [np.full(14, np.inf), np.full(14, -np.inf)] for k in ['state', 'action']}
    delta_sum = np.zeros(14)
    for root in roots:
        meta = json.loads((root / 'meta/info.json').read_text())
        episodes = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        root_frames = 0
        for episode in episodes:
            eid = episode['episode_index']
            file = root / meta['data_path'].format(episode_index=eid, episode_chunk=eid // meta['chunks_size'])
            columns = pq.read_table(file, columns=['observation.qpos', 'action', 'timestamp', 'episode_index'])
            state = np.asarray(columns['observation.qpos'].to_pylist(), dtype=np.float32)
            action = np.asarray(columns['action'].to_pylist(), dtype=np.float32)
            assert state.shape == action.shape == (episode['length'], 14), str(file)
            assert np.isfinite(state).all() and np.isfinite(action).all(), str(file)
            ts = columns['timestamp'].to_numpy()
            assert np.isfinite(ts).all() and np.all(np.diff(ts) > 0), str(file)
            assert np.all(columns['episode_index'].to_numpy() == eid), str(file)
            for key, values in [('state', state), ('action', action)]:
                extrema[key][0] = np.minimum(extrema[key][0], values.min(axis=0))
                extrema[key][1] = np.maximum(extrema[key][1], values.max(axis=0))
            delta_sum += np.abs(action - state).sum(axis=0, dtype=np.float64)
            root_frames += len(state)
        assert len(episodes) == meta['total_episodes']
        assert root_frames == meta['total_frames']
        entry['episodes'] += len(episodes)
        entry['frames'] += root_frames
        entry['roots'].append(str(root))
    entry['extrema'] = {k: {'min': v[0].tolist(), 'max': v[1].tolist()} for k,v in extrema.items()}
    entry['mean_abs_action_minus_state'] = (delta_sum / entry['frames']).tolist()
    report[task] = entry
    print('PASS', task, entry['episodes'], entry['frames'], flush=True)
output = Path('/report/numeric_audit.json')
with output.open('x') as stream:
    json.dump(report, stream, indent=2)
print('REPORT', output, flush=True)
