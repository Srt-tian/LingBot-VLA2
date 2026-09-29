"""Fail-fast real-loader acceptance; no random retry or norm/model required.

Run inside the validated image with PFS mounted. Samples first/middle/last frame
of every source, decodes current/future images, checks joint/gripper transforms.
This does not replace full normalization or forward/backward acceptance.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from lingbotvla.data.vla_data.base_dataset import VLADataset


def main():
    root = Path('/pfs/user/data/lingbot_vla_v2/piper_five_tasknorm')
    manifest = json.loads((root / 'manifest.json').read_text())
    arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    gripper = [6, 13]
    checked = 0
    for task, spec in manifest.items():
        for source in spec['views']:
            dataset = VLADataset(
                source, f'piper_{task}', SimpleNamespace(),
                str(root / 'robot_configs'), do_nomalize=False,
                return_item=True, disabled_image_features=False,
                use_future_image=True, chunk_size=50,
            )
            # Norm mode normally skips video; explicitly exercise the same
            # video decoder used in training without requiring norm/model.
            dataset.dataset.load_image = True
            for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
                raw = dataset.check_lerobot_item(dataset.dataset[index])
                assert raw['task'] == spec['prompt'], (task, source, raw['task'])
                state, action = raw['observation.qpos'], raw['action']
                assert tuple(state.shape) == (14,), state.shape
                assert tuple(action.shape) == (50, 14), action.shape
                expected_arm = action[:, arm] - state[arm]
                expected_gripper = action[:, gripper].clone()
                cameras = dataset.feature_transform.org_features['images']
                assert len(cameras) == 3, cameras
                for key in cameras:
                    value = raw[key]
                    assert value.ndim == 4 and value.shape[0] == 2
                    assert tuple(value.shape[-3:]) == (3, 224, 224), value.shape
                    assert torch.isfinite(value).all(), key
                transformed = dataset.feature_transform.apply(copy.deepcopy(raw))
                torch.testing.assert_close(transformed['action.arm.position'], expected_arm)
                torch.testing.assert_close(transformed['action.effector.position'], expected_gripper)
                torch.testing.assert_close(transformed['observation.state.arm.position'], state[arm])
                torch.testing.assert_close(transformed['observation.state.effector.position'], state[gripper])
                for key in dataset.action_features + dataset.state_features:
                    assert torch.isfinite(transformed[key]).all(), key
                checked += 1
            print(json.dumps({'task': task, 'source': source, 'status': 'PASS',
                              'frames': len(dataset)}, ensure_ascii=False), flush=True)
    print(f'REAL_LOADER_ACCEPTANCE_PASSED samples={checked}', flush=True)


if __name__ == '__main__':
    main()
