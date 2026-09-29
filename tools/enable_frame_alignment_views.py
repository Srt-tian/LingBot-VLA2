"""Opt in only audited isolated Piper metadata views; preserve original metadata."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/pfs/user/data/lingbot_vla_v2/piper_five_tasknorm')
audit_path = Path('/pfs/user/data/lingbot_vla_v2/preparation/frame_alignment_audit.json')
raw = audit_path.read_bytes()
audit = json.loads(raw)
assert audit['errors'] == []
assert sum(v['episodes'] for v in audit['views']) == 1676
assert sum(v['video_streams'] for v in audit['views']) == 5028
for view in audit['views']:
    path = Path(view['view']) / 'meta/info.json'
    assert path.resolve().is_relative_to(root / 'views') and not path.is_symlink()
    assert not path.with_suffix('.before_frame_alignment.json').exists()
for view in audit['views']:
    path = Path(view['view']) / 'meta/info.json'
    info = json.loads(path.read_text())
    assert info['total_frames'] == view['frames']
    shutil.copy2(path, path.with_suffix('.before_frame_alignment.json'))
    info['lingbot_frame_index_alignment'] = True
    info['lingbot_frame_alignment_audit_sha256'] = hashlib.sha256(raw).hexdigest()
    temporary = path.with_suffix('.alignment.tmp')
    with temporary.open('x') as stream:
        json.dump(info, stream, indent=2, ensure_ascii=False)
    temporary.replace(path)
print('AUDITED_FRAME_INDEX_ALIGNMENT_ENABLED views=14; RAW_SOURCES_UNCHANGED')
