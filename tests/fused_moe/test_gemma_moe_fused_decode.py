# SPDX-License-Identifier: Apache-2.0
"""Independent expert arithmetic, input contracts and current-stream replay."""

import pytest
import torch
import torch.nn.functional as F

from vllm_xpu_kernels.gemma_moe_fused_interface import (XpuGemmaMoEFusedDecode,
                                                        is_available,
                                                        workspace_bytes)

pytestmark = pytest.mark.skipif(not is_available(), reason="Gemma op not built")


@pytest.fixture(scope="module")
def expert_weights():
    torch.set_num_threads(4)
    g = torch.Generator().manual_seed(92310)

    def quant(shape):
        w = torch.randn(shape, generator=g) / 32
        scale = w.abs().amax(dim=(1, 2)) / 448
        return (w / scale[:, None, None]).to(torch.float8_e4m3fn), scale

    w13, s13 = quant((128, 2816, 704))
    w2, s2 = quant((128, 352, 2816))
    cpu = (w13, s13, w2, s2)
    gpu = tuple(t.to("xpu:0") for t in cpu)
    return cpu, gpu


def reference(x, parameters, weights, ids):
    w13, s13, w2, s2 = parameters
    partial = torch.zeros(x.shape[0], 8, 2816, dtype=torch.float32)
    for expert in ids.unique().tolist():
        if not 0 <= expert < 128:
            continue
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        gu = (x[rows].float() @ w13[expert].float() * s13[expert]).to(x.dtype)
        gate, up = gu.chunk(2, -1)
        activation = (
            F.gelu(gate.float(), approximate="tanh").to(x.dtype) * up
        ).to(x.dtype)
        partial[rows, slots] = (
            (activation.float() @ w2[expert].float() * s2[expert])
            .to(x.dtype)
            .float()
        )
    return (partial * weights[:, :, None]).sum(1).to(x.dtype)


def inputs(m, dtype, padding=False):
    g = torch.Generator().manual_seed(235 + m)
    x = (torch.randn(m, 2816, generator=g) * 0.5).to(dtype)
    ids = torch.rand(m, 128, generator=g).topk(8, -1).indices.int()
    weights = torch.softmax(torch.randn(m, 8, generator=g), -1)
    if padding:
        ids[:, 0] = -1
        ids[:, -1] = 128
    return x, weights, ids


@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_numeric_matches_independent_cpu_reference(expert_weights, m, dtype):
    cpu, gpu = expert_weights
    x, weights, ids = inputs(m, dtype)
    expected = reference(x, cpu, weights, ids)
    plan = XpuGemmaMoEFusedDecode(*gpu)
    x, weights, ids = (t.to("xpu:0") for t in (x, weights, ids))
    actual = plan.forward(torch.empty_like(x), x, weights, ids)
    torch.xpu.synchronize()
    tol = (1e-2, 2e-3) if dtype == torch.float16 else (3e-2, 1e-2)
    torch.testing.assert_close(actual.cpu(), expected, rtol=tol[0], atol=tol[1])


@pytest.mark.parametrize("m", [1, 4, 5, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_padding_ignores_invalid_weights(expert_weights, m, dtype):
    _, gpu = expert_weights
    x, weights, ids = inputs(m, dtype, padding=True)
    plan = XpuGemmaMoEFusedDecode(*gpu)
    x, weights, ids = (t.to("xpu:0") for t in (x, weights, ids))
    baseline = plan.forward(torch.empty_like(x), x, weights, ids).clone()
    weights[:, 0] = float("nan")
    weights[:, -1] = float("nan")
    actual = plan.forward(torch.empty_like(x), x, weights, ids)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    ids.fill_(-1)
    actual = plan.forward(actual, x, weights, ids)
    torch.xpu.synchronize()
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 5, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_current_stream_graph_replays_changed_inputs(expert_weights, m, dtype):
    _, gpu = expert_weights
    plan = XpuGemmaMoEFusedDecode(*gpu)
    x, weights, ids = (t.to("xpu:0") for t in inputs(m, dtype))
    output = torch.empty_like(x)
    stream = torch.xpu.Stream(device=0)
    with torch.xpu.stream(stream):
        for _ in range(2):
            plan.forward(output, x, weights, ids)
    stream.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        plan.forward(output, x, weights, ids)
    for delta in [0.125, -0.0625]:
        with torch.xpu.stream(stream):
            x.add_(delta)
            ids.copy_((ids + 11) % 128)
            graph.replay()
            captured = output.clone()
            eager = plan.forward(torch.empty_like(x), x, weights, ids)
        stream.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "alias",
        "workspace",
        "workspace_overlap",
        "m9",
        "int64_ids",
        "strided",
        "misaligned",
    ],
)
def test_rejects_before_output_write(expert_weights, case):
    _, gpu = expert_weights
    m = 9 if case == "m9" else 1
    x, weights, ids = (t.to("xpu:0") for t in inputs(m, torch.float16))
    output = torch.full_like(x, 3)
    ws = torch.zeros(
        workspace_bytes(min(m, 8)), device="xpu:0", dtype=torch.uint8
    )
    if case == "alias":
        output = x
    if case == "workspace":
        ws = ws[:1]
    if case == "workspace_overlap":
        output = ws[: 2816 * 2].view(torch.float16).view(1, 2816)
    if case == "int64_ids":
        ids = ids.long()
    if case == "strided":
        x = torch.zeros(1, 5632, device="xpu:0", dtype=x.dtype)[:, ::2]
    if case == "misaligned":
        x = torch.zeros(2817, device="xpu:0", dtype=x.dtype)[1:].view(1, 2816)
    before = output.clone()
    with pytest.raises(RuntimeError):
        torch.ops._xpu_C.gemma_moe_fused_decode_interface(
            output, x, weights, ids, *gpu, ws
        )
    torch.xpu.synchronize()
    torch.testing.assert_close(output, before, rtol=0, atol=0)


def test_workspace_bound_to_weights_device(expert_weights):
    _, gpu = expert_weights
    plan = XpuGemmaMoEFusedDecode(*gpu)
    ws = plan.workspace(1)
    with torch.xpu.device(1):
        assert plan.workspace(1) is ws
        assert plan.workspace(5).device == gpu[0].device


@pytest.mark.parametrize("m", [0, 9, -1])
def test_bad_workspace_count_rejected(m):
    with pytest.raises(ValueError):
        workspace_bytes(m)


@pytest.mark.parametrize("index", range(9))
@pytest.mark.parametrize("destination", ["cpu", "xpu:1"])
def test_every_argument_rejects_foreign_device(
    expert_weights, index, destination
):
    _, parameters = expert_weights
    x, weights, ids = (t.to("xpu:0") for t in inputs(1, torch.float16))
    output = torch.full_like(x, 3)
    workspace = torch.zeros(
        workspace_bytes(1), device="xpu:0", dtype=torch.uint8
    )
    args = [output, x, weights, ids, *parameters, workspace]
    args[index] = args[index].to(destination)
    before = args[0].clone()
    with pytest.raises(RuntimeError):
        torch.ops._xpu_C.gemma_moe_fused_decode_interface(*args)
    torch.xpu.synchronize()
    torch.testing.assert_close(args[0], before, rtol=0, atol=0)


def routing_reference(logits, scales):
    values = logits.float()
    bits = values.view(torch.int32)
    key = torch.where(bits < 0, bits ^ -2147483648, bits ^ -1)
    packed = ((key.long() & 0xFFFFFFFF) << 32) | torch.arange(128)
    ids = packed.argsort(dim=-1)[:, :8].int()
    selected = values.gather(1, ids.long())
    raw = torch.exp2(
        (selected - values.amax(1, keepdim=True)) * 1.4426950408889634
    )
    weights = raw / raw.sum(1, keepdim=True) * scales.float()[ids.long()]
    return weights, ids


@pytest.mark.parametrize("m", [1, 4, 5, 8])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_routing_matches_cpu_math_and_tie_order(m, dtype):
    g = torch.Generator().manual_seed(917 + m)
    logits = torch.randn(m, 128, generator=g).to(dtype)
    logits[:, :16] = 4  # Equal maxima must select ascending expert IDs.
    scales = (torch.rand(128, generator=g) + 0.5).to(dtype)
    expected_weights, expected_ids = routing_reference(logits, scales)
    weights, ids = torch.ops._xpu_C.gemma_moe_topk_interface(
        logits.xpu(), scales.xpu(), 8
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        weights.cpu(), expected_weights, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("pattern", ["random", "signed_zero", "large"])
@pytest.mark.parametrize(
    "scale_dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_routing_fp32_logits_with_active_scale_dtypes(pattern, scale_dtype):
    g = torch.Generator().manual_seed(7123)
    logits = torch.randn(8, 128, generator=g)
    if pattern == "signed_zero":
        logits.zero_()
        logits[:, 1::2] = -0.0
    if pattern == "large":
        logits *= 1000
    scales = (torch.rand(128, generator=g) + 0.5).to(scale_dtype)
    ew, ei = routing_reference(logits, scales)
    w, ids = torch.ops._xpu_C.gemma_moe_topk_interface(
        logits.xpu(), scales.xpu(), 8
    )
    torch.xpu.synchronize()
    torch.testing.assert_close(ids.cpu(), ei, rtol=0, atol=0)
    torch.testing.assert_close(w.cpu(), ew, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "case",
    [
        "m0",
        "m9",
        "experts",
        "topk",
        "logits_dtype",
        "scale_dtype",
        "scale_shape",
        "strided",
        "cpu_logits",
        "cpu_scale",
        "foreign_scale",
    ],
)
def test_routing_rejects_unsupported_inputs(case):
    logits = torch.zeros(1, 128, device="xpu:0")
    scale = torch.ones(128, device="xpu:0")
    topk = 8
    if case in ("m0", "m9"):
        logits = torch.zeros(0 if case == "m0" else 9, 128, device="xpu:0")
    if case == "experts":
        logits = logits[:, :127].contiguous()
    if case == "topk":
        topk = 7
    if case == "logits_dtype":
        logits = logits.double()
    if case == "scale_dtype":
        scale = scale.double()
    if case == "scale_shape":
        scale = scale.reshape(1, 128)
    if case == "strided":
        scale = torch.ones(256, device="xpu:0")[::2]
    if case == "cpu_logits":
        logits = logits.cpu()
    if case == "cpu_scale":
        scale = scale.cpu()
    if case == "foreign_scale":
        scale = scale.to("xpu:1")
    with pytest.raises(RuntimeError):
        torch.ops._xpu_C.gemma_moe_topk_interface(logits, scale, topk)


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16])
def test_routing_current_stream_graph_replays_active_scales(m, scale_dtype):
    logits = torch.randn(m, 128, device="xpu:0")
    scales = torch.ones(128, device="xpu:0", dtype=scale_dtype)
    stream = torch.xpu.Stream(device=0)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.stream(stream):
        torch.ops._xpu_C.gemma_moe_topk_interface(logits, scales, 8)
    stream.synchronize()
    with torch.xpu.graph(graph, stream=stream):
        weights, ids = torch.ops._xpu_C.gemma_moe_topk_interface(
            logits, scales, 8
        )
    for shift in [1.0, -0.5]:
        with torch.xpu.stream(stream):
            logits.copy_(torch.roll(logits, 11, dims=1))
            scales.add_(shift)
            graph.replay()
            actual_w, actual_i = weights.clone(), ids.clone()
        stream.synchronize()
        ew, ei = routing_reference(logits.cpu(), scales.cpu())
        torch.testing.assert_close(actual_i.cpu(), ei, rtol=0, atol=0)
        torch.testing.assert_close(actual_w.cpu(), ew, rtol=1e-5, atol=1e-6)
