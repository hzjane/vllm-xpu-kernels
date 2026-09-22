# SPDX-License-Identifier: Apache-2.0
"""Numerical regression for the existing and extended small-M decode paths."""

import pytest
import torch

import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401


@pytest.mark.parametrize("m", [1, 2, 5, 8])
@pytest.mark.parametrize("width", [256, 512, 2816, 5376])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_rms_decode_strided(m, width, weighted, offset):
    torch.manual_seed(831)
    heads = 8 if width == 256 else 1
    storage = torch.randn(m, heads, width + 8).half().to("xpu")
    x = storage[..., offset:offset + width]
    weight = torch.randn(width).half().to("xpu") if weighted else None
    ref = x.cpu().float()
    ref = (ref * torch.rsqrt(ref.square().mean(-1, keepdim=True)
                             + 1e-6))
    if weight is not None:
        ref = ref * weight.cpu().float()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(y, x, weight, 1e-6)
    torch.testing.assert_close(y.cpu(), ref.half(), rtol=1e-3, atol=2e-3)
