# SPDX-License-Identifier: Apache-2.0
"""Gemma fusion round points, partial RoPE and current-stream contracts."""

import pytest
import torch

from vllm_xpu_kernels import gemma4_router_preprocess  # noqa: F401


def norm(x, weight, eps):
    out = (
        x.float() * (x.float().square().mean(-1, keepdim=True) + eps).rsqrt()
    ).half()
    return out if weight is None else out * weight


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 9])
@pytest.mark.parametrize("strided", [False, True])
def test_router_preserves_preprojection_rounding(rows, strided):
    torch.manual_seed(55)
    x = torch.randn(
        rows * 2 if strided else rows, 2816, device="xpu", dtype=torch.float16
    )
    if strided:
        x = x[::2]
    root = torch.tensor(2816**-0.5, device="xpu", dtype=x.dtype)
    scale = torch.randn(2816, device="xpu", dtype=x.dtype)
    expected = norm(x, None, 1e-6) * root * scale
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        actual = torch.ops._xpu_C.gemma4_router_preprocess(x, root, scale, 1e-6)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=2e-3)


@pytest.mark.parametrize("rows", [1, 2, 5, 8])
@pytest.mark.parametrize(
    "qh,kh,d,rd",
    [
        (8, 4, 256, 256),
        (16, 8, 256, 256),
        (8, 1, 512, 128),
        (16, 2, 512, 128),
        (8, 1, 512, 512),
        (16, 2, 512, 512),
    ],
)
@pytest.mark.parametrize("native", [False, True])
def test_qkv_norm_and_full_partial_rope(rows, qh, kh, d, rd, native):
    torch.manual_seed(43)
    x = torch.randn(rows, (qh + 2 * kh) * d, device="xpu", dtype=torch.float16)
    qw = torch.randn(d, device="xpu", dtype=x.dtype)
    kw = torch.randn(d, device="xpu", dtype=x.dtype)
    phase = torch.randn(128, rd // 2, device="xpu")
    if d == rd == 512:
        phase[:, 64:] = 0  # Gemma proportional identity frequencies.
    cache = torch.cat([phase.cos(), phase.sin()], -1).half()
    positions = torch.arange(rows, device="xpu", dtype=torch.int64) * 7
    q, k, v = x.split([qh * d, kh * d, kh * d], -1)
    q = norm(q.reshape(rows, qh, d), qw, 1e-6)
    k = norm(k.reshape(rows, kh, d), kw, 1e-6)
    v = norm(v.reshape(rows, kh, d), None, 1e-6)
    c, s = cache[positions].chunk(2, -1)

    def rope(x):
        left, right = x[..., :rd].chunk(2, -1)
        if native:
            out = torch.cat(
                [
                    left * c[:, None] - right * s[:, None],
                    right * c[:, None] + left * s[:, None],
                ],
                -1,
            )
        else:
            out = torch.cat(
                [
                    left.float() * c[:, None].float()
                    - right.float() * s[:, None].float(),
                    right.float() * c[:, None].float()
                    + left.float() * s[:, None].float(),
                ],
                -1,
            ).half()
        return torch.cat([out, x[..., rd:]], -1).reshape(rows, -1)

    expected = (rope(q), rope(k), v.reshape(rows, -1))
    actual = torch.ops._xpu_C.gemma4_qkv_norm_rope(
        x, positions, qw, kw, cache, qh, kh, d, 1e-6, native
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=4e-3, rtol=3e-3)
    # The source QKV is functional and never modified.
    assert all(a.data_ptr() != x.data_ptr() for a in actual)


def test_router_rejects_root_that_changes_output_rank():
    x = torch.ones(1, 2816, device="xpu", dtype=torch.float16)
    scale = torch.ones(2816, device="xpu", dtype=x.dtype)
    root = torch.ones(1, 1, 1, device="xpu", dtype=x.dtype)
    with pytest.raises(RuntimeError, match="scalar root"):
        torch.ops._xpu_C.gemma4_router_preprocess(x, root, scale, 1e-6)
