"""CPU forward/backward equivalence for real Qwen3-VL packed vision attention."""
import copy
import importlib.util
from types import SimpleNamespace
import torch
from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import Qwen3VLVisionAttention

assert importlib.util.find_spec('flash_attn') is None
torch.manual_seed(7)
config = SimpleNamespace(hidden_size=32, num_heads=4, _attn_implementation='eager')
reference = Qwen3VLVisionAttention(config)
candidate = copy.deepcopy(reference)
candidate.config._attn_implementation = 'sdpa'
x = torch.randn(6, 32, requires_grad=True)
y = x.detach().clone().requires_grad_(True)
segments = torch.tensor([0, 2, 6], dtype=torch.int32)
position = (torch.ones(6, 8), torch.zeros(6, 8))
a = reference(x, segments, position_embeddings=position)
b = candidate(y, segments, position_embeddings=position)
assert a.shape == b.shape == (6, 32)
torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
a.square().mean().backward()
b.square().mean().backward()
torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
for (name, p), (other, q) in zip(reference.named_parameters(), candidate.named_parameters()):
    assert name == other and q.grad is not None and torch.isfinite(q.grad).all()
    torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
with torch.no_grad():
    changed = y.detach().clone()
    changed[:2] += 100
    c = candidate(changed, segments, position_embeddings=position)
    torch.testing.assert_close(b[2:], c[2:], atol=2e-6, rtol=2e-5)
print('SDPA_VISION_FORWARD_BACKWARD_AND_SEGMENT_ISOLATION_PASSED')
