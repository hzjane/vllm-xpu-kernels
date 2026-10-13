# SPDX-License-Identifier: Apache-2.0
"""Independent math and stream checks for the Gemma compound decode ops."""

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU required"
)


def rms(x, w, eps):
    normalized = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    ).half()
    return normalized if w is None else normalized * w


def router_inputs(m, seed=23):
    torch.manual_seed(seed)
    x = torch.randn(m, 2816, device="xpu", dtype=torch.float16)
    scale = torch.randn(2816, device="xpu", dtype=torch.float16) * 0.1 + 1
    root = torch.tensor(2816**-0.5, device="xpu", dtype=torch.float32)
    projection = (
        torch.randn(128, 2816, device="xpu", dtype=torch.float16) * 0.03
    )
    mw = torch.randn_like(scale) * 0.1 + 1
    expert_scale = torch.rand(128, device="xpu", dtype=torch.float32) + 0.2
    return x, scale, root, projection, mw, expert_scale


@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("residual", [False, True])
def test_router_fusion_math(m, residual):
    x, scale, root, projection, mw, expert_scale = router_inputs(m)
    if residual:
        r = torch.randn_like(x)
        pw = torch.randn_like(scale) * 0.1 + 1
        dw = torch.randn_like(scale) * 0.1 + 1
        dense, moe, logits, weights, ids, added = (
            torch.ops._xpu_C.gemma4_residual_norm_router_topk(
                x,
                r,
                pw,
                dw,
                scale,
                root,
                projection,
                mw,
                expert_scale,
                1e-6,
                2e-6,
                3e-6,
                4e-6,
            )
        )
        source = rms(x, pw, 1e-6) + r
        torch.testing.assert_close(added, source, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(
            dense, rms(source, dw, 2e-6), rtol=3e-3, atol=3e-3
        )
    else:
        source = x
        logits, moe, weights, ids = torch.ops._xpu_C.gemma4_norm_router_topk(
            x,
            scale,
            root,
            projection,
            mw,
            expert_scale,
            3e-6,
            4e-6,
        )
    router_x = (rms(source, None, 3e-6) * root.half()) * scale
    reference_logits = F.linear(router_x, projection).float()
    torch.testing.assert_close(logits, reference_logits, rtol=7e-3, atol=7e-5)
    torch.testing.assert_close(moe, rms(source, mw, 4e-6), rtol=3e-3, atol=3e-3)
    # Check selection independently from projection's permissible FP16 rounding.
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :8]
    selected = logits.gather(-1, order)
    ref_weights = torch.softmax(selected, -1) * expert_scale[order]
    torch.testing.assert_close(ids.long(), order, rtol=0, atol=0)
    torch.testing.assert_close(weights, ref_weights, rtol=2e-5, atol=2e-6)


def test_router_ties_and_streams():
    tensors = list(router_inputs(3))
    tensors[3] = torch.zeros_like(tensors[3])
    tensors[-1] = tensors[-1].half()
    torch.xpu.synchronize()
    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    outputs = []
    for stream in streams:
        with torch.xpu.stream(stream):
            outputs.append(
                torch.ops._xpu_C.gemma4_norm_router_topk(*tensors, 1e-6, 1e-6)
            )
    for stream in streams:
        stream.synchronize()
    for logits, moe, weights, ids in outputs:
        expected = torch.arange(8, device="xpu", dtype=torch.int32).expand(
            3, -1
        )
        torch.testing.assert_close(ids, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            weights, (tensors[-1][:8].float() / 8).expand(3, -1)
        )
    assert outputs[0][1].data_ptr() != outputs[1][1].data_ptr()


def dense_routed_inputs(m):
    torch.manual_seed(29)
    x = torch.randn(m, 2816, device="xpu", dtype=torch.float16) * 0.3
    dense_x = torch.randn_like(x)
    w13 = torch.randn(128, 2816, 704, device="xpu").to(torch.float8_e4m3fn)
    w2 = torch.randn(128, 352, 2816, device="xpu").to(torch.float8_e4m3fn)
    s13 = torch.full((128,), 0.015, device="xpu")
    s2 = torch.full((128,), 0.015, device="xpu")
    gate = torch.randn(2112, 2816, device="xpu").to(torch.float8_e4m3fn).T
    down = torch.randn(2816, 1056, device="xpu").to(torch.float8_e4m3fn).T
    scale = torch.tensor([0.015], device="xpu")
    ids = torch.arange(m * 8, device="xpu", dtype=torch.int32).view(m, 8) % 128
    weights = torch.softmax(torch.randn(m, 8, device="xpu"), -1)
    return dense_x, x, gate, scale, down, scale, w13, s13, w2, s2, weights, ids


@pytest.mark.parametrize("m", [1, 2, 4])
def test_dense_routed_shared_grids(m):
    args = dense_routed_inputs(m)
    dense_out, routed_out = torch.empty_like(args[0]), torch.empty_like(args[1])
    assert torch.ops._xpu_C.fp8_moe_dense_decode(dense_out, routed_out, *args)
    routed_ref = torch.empty_like(routed_out)
    assert torch.ops._xpu_C.fp8_moe_decode(routed_ref, args[1], *args[6:])
    gate_up = ((args[0].float() @ args[2].float()) * args[3]).half()
    gate, up = gate_up.chunk(2, -1)
    activation = F.gelu(gate, approximate="tanh") * up
    dense_ref = ((activation.float() @ args[4].float()) * args[5]).half()
    torch.testing.assert_close(routed_out, routed_ref, rtol=0, atol=0)
    torch.testing.assert_close(dense_out, dense_ref, rtol=3e-2, atol=3e-3)


def test_dense_routed_rejects_before_writing():
    args = dense_routed_inputs(5)
    dense_out = torch.full_like(args[0], 123)
    routed_out = torch.full_like(args[1], -456)
    assert not torch.ops._xpu_C.fp8_moe_dense_decode(
        dense_out, routed_out, *args
    )
    assert (dense_out == 123).all() and (routed_out == -456).all()
    args = tuple(
        x[:1] if i in (0, 1, 10, 11) else x for i, x in enumerate(args)
    )
    assert not torch.ops._xpu_C.fp8_moe_dense_decode(
        args[0], torch.empty_like(args[1]), *args
    )
