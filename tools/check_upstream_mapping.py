"""Execute upstream packing functions in isolation, without model/GPU imports.

This is a unit test, not a substitute for the real dataset loader smoke.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

root = Path('/source/lingbotvla/data/vla_data')
parsed = ast.parse((root / 'utils.py').read_text())
cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'FeatureTransform')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'pad_and_concat')
namespace = {'torch': torch, 'F': F}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(root / 'utils.py'), 'exec'), namespace)
for name in ['prepare_state', 'prepare_action']:
    parsed = ast.parse((root / 'transform.py').read_text())
    func = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    func.returns = None
    for arg in func.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(root / 'transform.py'), 'exec'), namespace)

dims = {'arm.position': 14, 'end.position': 14, 'effector.position': 2,
        'waist.position': 4, 'head.position': 2, 'base.position': 3, 'hand.position': 12}
obj = SimpleNamespace(feature_config=SimpleNamespace(joints=list(dims), joints_max_dim=dims, images=[]),
                      states=['observation.state.arm.position', 'observation.state.effector.position'],
                      actions=['action.arm.position', 'action.effector.position'], use_future_image=False, chunk_size=50)
item = {'observation.state.arm.position': torch.arange(12, dtype=torch.float32),
        'observation.state.effector.position': torch.tensor([.02, .06]),
        'action.arm.position': torch.arange(12, dtype=torch.float32).repeat(50, 1),
        'action.effector.position': torch.tensor([.03, .07]).repeat(50, 1),
        'action_is_pad': torch.zeros(50, dtype=torch.bool), 'task': 'test'}
packed = namespace['pad_and_concat'](obj, item, True)
state = namespace['prepare_state'](packed, 55)
action = namespace['prepare_action'](packed, 55)
mask = F.pad(packed['action_joint_mask'], (0, 55-len(packed['action_joint_mask'])))
expected = torch.zeros(55, dtype=torch.bool)
expected[:12] = True
expected[28:30] = True
assert state.shape == (55,) and action.shape == (50,55)
assert torch.equal(mask, expected), mask
assert torch.count_nonzero(state[~mask]) == 0
assert torch.count_nonzero(action[:,~mask]) == 0
assert torch.equal(state[:12], item['observation.state.arm.position'])
assert torch.equal(action[:,28:30], item['action.effector.position'])
print('PASS upstream packing: state55/action50x55; supervised indices0..11,28,29; all inactive slots zero', flush=True)
