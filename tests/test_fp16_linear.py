# SPDX-License-Identifier: Apache-2.0
"""仅保留 Router 后的数学、当前 stream 和原 Linear 回退契约。"""
import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels  # noqa: F401


@pytest.fixture
def linear_op():
    return torch.ops._xpu_C.unquantized_gemm


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 9])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_router_math_and_current_stream(linear_op, rows, bias):
    generator = torch.Generator().manual_seed(18)
    x = torch.randn(rows, 2816, generator=generator).half() * 0.1
    w = torch.randn(128, 2816, generator=generator).half() * 0.1
    b = torch.randn(128, generator=generator).half() if bias else None
    expected = F.linear(
        x.float(), w.float(), b.float() if b is not None else None
    ).half()
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        actual = linear_op(
            x.xpu(), w.xpu(), b.xpu() if b is not None else None
        ).clone()
    stream.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("rows,n,k", [(1, 2816, 512), (5, 512, 5632),
                                     (1, 8192, 1024), (1, 128, 5376)])
@torch.inference_mode()
def test_non_router_shapes_use_native_linear(linear_op, rows, n, k):
    torch.manual_seed(27)
    x = torch.randn(rows, k, device="xpu", dtype=torch.float16) * 0.1
    w = torch.randn(n, k, device="xpu", dtype=x.dtype) * 0.1
    torch.testing.assert_close(linear_op(x, w), F.linear(x, w), atol=0, rtol=0)


@torch.inference_mode()
def test_strided_rows_preserve_native_math(linear_op):
    # Row stride remains aligned; the fast path requires contiguous storage.
    torch.manual_seed(28)
    x = torch.randn(6, 2816).half()[::2] * 0.1
    storage = torch.randn(6, 2816, device="xpu", dtype=torch.float16)
    device_x = storage[::2]
    device_x.copy_(x)
    w = torch.randn(128, 2816).half() * 0.1
    expected = F.linear(x.float(), w.float()).half()
    actual = linear_op(device_x, w.xpu())
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("shape,n", [((1, 2816), 128),
                                     ((1, 2816), 131072), ((2, 3, 64), 16)])
@pytest.mark.parametrize("bias", [False, True])
def test_meta_preserves_linear_contract(linear_op, shape, n, bias):
    x = torch.empty(shape, device="meta", dtype=torch.float16)
    w = torch.empty(n, shape[-1], device="meta", dtype=x.dtype)
    b = torch.empty(n, device="meta", dtype=x.dtype) if bias else None
    actual = linear_op(x, w, b)
    expected = F.linear(x, w, b)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
