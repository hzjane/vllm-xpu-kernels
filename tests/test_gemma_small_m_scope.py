# SPDX-License-Identifier: Apache-2.0
"""Gemma TP2 dispatch boundaries, independent math and no-write rejection."""
import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._C  # noqa: F401


def rope_inputs(m, qh, kh, head, rotary, layout, dtype=torch.float16):
    torch.manual_seed(92319)
    width = (qh + 2 * kh) * head
    source = torch.randn(m, width, dtype=dtype, device="xpu")
    query = source[:, :qh * head]
    key = source[:, qh * head:(qh + kh) * head]
    if layout == "contiguous":
        query, key = query.clone(), key.clone()
    elif layout == "padded":
        query = torch.randn(m, qh * head + 32, dtype=dtype,
                            device="xpu")[:, :qh * head]
    cache = torch.randn(32, rotary, dtype=dtype, device="xpu")
    pos = torch.arange(m, dtype=torch.int64, device="xpu") + 3
    return pos, query, key, head, cache


def reference_rope(data, positions, head, cache):
    m = data.shape[0]
    result = data.cpu().reshape(m, -1, head).clone()
    rotary = cache.shape[1]
    half = rotary // 2
    cs = cache.cpu()[positions.cpu()].unsqueeze(1)
    a, b = result[..., :half].clone(), result[..., half:rotary].clone()
    result[..., :half] = a * cs[..., :half] - b * cs[..., half:]
    result[..., half:rotary] = b * cs[..., :half] + a * cs[..., half:]
    return result.reshape(data.shape)


@pytest.mark.parametrize("m", [1, 5, 8])
@pytest.mark.parametrize("layout", ["contiguous", "qkv"])
@pytest.mark.parametrize("qh,kh,head,rotary", [
    (8, 4, 256, 256), (16, 8, 256, 256),
    (8, 1, 512, 512), (16, 2, 512, 512),
])
def test_gemma_rope_current_stream(m, layout, qh, kh, head, rotary):
    args = rope_inputs(m, qh, kh, head, rotary, layout)
    pos, q, k, head, cache = args
    expected = [reference_rope(x, pos, head, cache) for x in (q, k)]
    torch.xpu.synchronize()
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        assert torch.ops._xpu_C.try_rotary_embedding_small_m(*args)
        actual = [q.clone(), k.clone()]
    stream.synchronize()
    for result, ref in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("m,qh,kh,head,rotary,layout,dtype", [
    (9, 8, 4, 256, 256, "contiguous", torch.float16),
    (5, 8, 1, 256, 256, "contiguous", torch.float16),  # Qwen TP2
    (5, 16, 8, 512, 128, "contiguous", torch.float16),
    (5, 8, 1, 512, 128, "contiguous", torch.float16),
    (5, 8, 4, 256, 128, "contiguous", torch.float16),
    (5, 8, 4, 256, 256, "padded", torch.float16),
    (5, 8, 4, 256, 256, "contiguous", torch.bfloat16),
])
def test_other_rope_inputs_reject_without_writing(
        m, qh, kh, head, rotary, layout, dtype):
    args = rope_inputs(m, qh, kh, head, rotary, layout, dtype)
    pos, q, k, head, cache = args
    before = [q.clone(), k.clone()]
    assert not torch.ops._xpu_C.try_rotary_embedding_small_m(*args)
    for result, ref in zip((q, k), before):
        torch.testing.assert_close(result, ref, rtol=0, atol=0)


@pytest.mark.parametrize("rows,width,specialized", [
    (1, 1056, True), (8, 1056, True), (9, 1056, False),
    (64, 1056, False), (8, 352, True), (40, 352, True),
    (64, 352, True), (1, 352, False), (7, 352, False),
    (9, 352, False), (65, 352, False),
])
def test_shared_and_routed_gelu_boundaries(rows, width, specialized):
    torch.manual_seed(51)
    x = torch.randn(rows, 2 * width).half()
    expected = F.gelu(x[:, :width].float(), approximate="tanh").half()
    expected = expected * x[:, width:]
    device_x = x.xpu()
    out = torch.full((rows, width), 123., dtype=x.dtype, device="xpu")
    if specialized:
        torch.ops._xpu_C.gelu_tanh_and_mul_small_m(out, device_x)
    else:
        with pytest.raises(RuntimeError, match="Expected Gemma TP2"):
            torch.ops._xpu_C.gelu_tanh_and_mul_small_m(out, device_x)
        assert torch.all(out == 123.).item()
    torch.ops._C.gelu_tanh_and_mul(out, device_x)
    torch.testing.assert_close(out.cpu(), expected, atol=2e-3, rtol=2e-3)
