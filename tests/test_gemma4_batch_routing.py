# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_xpu_kernels import _moe_C  # noqa: F401


def reference(logits, scale):
    values = logits.float()
    bits = values.view(torch.int32).to(torch.int64) & 0xffffffff
    keys = torch.where(bits & 0x80000000 != 0,
                       bits ^ 0x80000000, bits ^ 0xffffffff)
    experts = torch.arange(128, device=logits.device)
    packed = (keys << 32) | experts
    ids = packed.argsort(dim=-1)[:, :8]
    maximum = torch.where(values.isnan(), -torch.inf, values).amax(
        -1, keepdim=True)
    exps = torch.exp2((values.gather(1, ids)-maximum) * 1.4426950408889634)
    total = exps.sum(-1, keepdim=True)
    weights = exps / torch.where(total > 0, total, 1)
    if scale is not None:
        weights = weights * scale[ids].float()
    return weights, ids.int()


@pytest.mark.parametrize('m', [1, 7, 9, 31, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('scale_dtype', [None, torch.float16, torch.float32])
@pytest.mark.parametrize('pattern', ['random', 'ties', 'nonfinite'])
def test_batch_routing(m, dtype, scale_dtype, pattern):
    torch.manual_seed(9871)
    x = torch.randn(m, 128, dtype=dtype, device='xpu')
    if pattern == 'ties':
        x[:, :32] = 2
        x[:, 32:64] = 0
        x[:, 64:96] = -0.0
    elif pattern == 'nonfinite':
        x[:, 1] = torch.nan
        x[:, 3] = -torch.nan
        x[:, 17] = torch.inf
        x[:, 19] = -torch.inf
        x[0, :] = -torch.inf
    scale = None if scale_dtype is None else (
        torch.rand(128, device='xpu', dtype=scale_dtype) + 0.5)
    expected_w, expected_ids = reference(x, scale)
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        w, ids = torch.ops._moe_C.gemma4_batch_topk(x, scale, 8)
    torch.xpu.current_stream().wait_stream(stream)
    torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        w, expected_w, atol=2e-6, rtol=2e-5, equal_nan=True)


def test_reject_invalid_inputs_and_meta():
    x = torch.zeros(256, 128, device='xpu')
    scale = torch.ones(128, device='xpu')
    cases = [(x[:, ::2], scale, 8), (x, scale.cpu(), 8),
             (x, scale[:64], 8), (x, scale, 7), (x.double(), scale, 8),
             (x[:0], scale, 8), (torch.empty(2049,128,device='xpu'), scale, 8)]
    for args in cases:
        with pytest.raises(RuntimeError):
            torch.ops._moe_C.gemma4_batch_topk(*args)
    w, ids = torch.ops._moe_C.gemma4_batch_topk(x.to('meta'), None, 8)
    assert w.shape == ids.shape == (256,8)
    assert w.dtype == torch.float32 and ids.dtype == torch.int32
