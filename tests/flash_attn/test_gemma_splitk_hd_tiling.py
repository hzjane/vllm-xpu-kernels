# SPDX-License-Identifier: Apache-2.0
"""Gemma decode geometries: head-dimension tiling versus FP32 attention."""

import pytest
import torch

from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU required"
)


@pytest.mark.parametrize(
    "heads,kv_heads,dim", [(8, 4, 256), (16, 1, 512), (16, 2, 512)]
)
@pytest.mark.parametrize("length", [4097, 32768])
def test_gemma_splitk_tiled_math_and_lse(
    monkeypatch, heads, kv_heads, dim, length
):
    monkeypatch.setenv("VLLM_XPU_GEMMA_SPLITK_HD_TILING", "1")
    torch.manual_seed(41)
    blocks = (length + 63) // 64
    q = torch.randn(1, heads, dim, device="xpu", dtype=torch.float16)
    k = torch.randn(
        blocks, 64, kv_heads, dim, device="xpu", dtype=torch.float16
    )
    v = torch.randn_like(k)
    table = torch.arange(blocks, device="xpu", dtype=torch.int32).view(
        1, blocks
    )
    cu_q = torch.tensor([0, 1], device="xpu", dtype=torch.int32)
    used_k = torch.tensor([length], device="xpu", dtype=torch.int32)
    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=1,
        cu_seqlens_q=cu_q,
        max_seqlen_k=length,
        seqused_k=used_k,
        block_table=table,
        causal=True,
        softmax_scale=dim**-0.5,
        num_splits=0,
        return_softmax_lse=True,
    )
    keys = (
        k.flatten(0, 1)[:length].float().repeat_interleave(heads // kv_heads, 1)
    )
    values = (
        v.flatten(0, 1)[:length].float().repeat_interleave(heads // kv_heads, 1)
    )
    scores = torch.einsum("qhd,khd->hqk", q.float(), keys) * dim**-0.5
    ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).half()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(
        lse.reshape(heads, 1), scores.logsumexp(-1), rtol=2e-3, atol=2e-3
    )
