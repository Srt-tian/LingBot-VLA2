"""Read-only audit of every episode's row indices and three video stream headers."""
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq

root = Path('/pfs/user/data/lingbot_vla_v2/piper_five_tasknorm')
manifest = json.loads((root / 'manifest.json').read_text())
report = {'views': [], 'errors': [], 'note': 'Header frame counts do not establish physical sensor synchrony.'}
for task, spec in manifest.items():
    for view_name in spec['views']:
        view = Path(view_name)
        info = json.loads((view / 'meta/info.json').read_text())
        episodes = [json.loads(x) for x in (view / 'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
        cameras = [k for k,v in info['features'].items() if v['dtype'] == 'video']
        result = {'task': task, 'view': view_name, 'episodes': 0, 'frames': 0,
                  'video_streams': 0, 'irregular_intervals': 0,
                  'max_timestamp_vs_index_error_s': 0., 'min_fps': 1e9, 'max_fps': 0.}
        for episode in episodes:
            idx, count = episode['episode_index'], episode['length']
            kw = dict(episode_chunk=idx // info.get('chunks_size', 1000), episode_index=idx)
            data = pq.read_table(view / info['data_path'].format(**kw), columns=['timestamp', 'frame_index'])
            ts = np.asarray(data['timestamp'], dtype=np.float64)
            fi = np.asarray(data['frame_index'])
            if len(fi) != count or not np.array_equal(fi, np.arange(count)):
                report['errors'].append([view_name, idx, 'noncontiguous_frame_index'])
            if len(ts) > 1 and not np.all(np.diff(ts) > 0):
                report['errors'].append([view_name, idx, 'nonmonotonic_timestamp'])
            result['irregular_intervals'] += int(np.sum(np.abs(np.diff(ts)-1/info['fps']) > 1e-4))
            result['max_timestamp_vs_index_error_s'] = max(result['max_timestamp_vs_index_error_s'], float(np.max(np.abs(ts-fi/info['fps']))))
            for camera in cameras:
                path = view / info['video_path'].format(**kw, video_key=camera)
                with av.open(str(path)) as video:
                    stream = video.streams.video[0]
                    frames = stream.frames
                    fps = float(stream.average_rate)
                    if frames != count:
                        report['errors'].append([view_name, idx, camera, 'frame_count', frames, count])
                    if abs(fps-info['fps']) > 0.1:
                        report['errors'].append([view_name, idx, camera, 'fps', fps])
                    result['min_fps'] = min(result['min_fps'], fps)
                    result['max_fps'] = max(result['max_fps'], fps)
                result['video_streams'] += 1
            result['episodes'] += 1
            result['frames'] += count
        report['views'].append(result)
        print(json.dumps(result), flush=True)
output = Path('/pfs/user/data/lingbot_vla_v2/preparation/frame_alignment_audit.json')
with output.open('x') as stream:
    json.dump(report, stream, indent=2)
print('AUDIT_FINISHED errors=' + str(len(report['errors'])), flush=True)
if report['errors']:
    raise SystemExit(1)
