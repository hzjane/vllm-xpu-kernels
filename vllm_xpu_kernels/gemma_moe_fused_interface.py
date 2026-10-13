# SPDX-License-Identifier: Apache-2.0
"""Routed per-tensor FP8 GELU-tanh experts for Gemma4 small-M decode.

The caller supplies Gemma routing weights, including its per-expert scale.
Invalid expert IDs contribute zero. Dense MLP, normalization and TP reduction
remain with the caller. Each plan owns scratch on the weights device; calls
sharing scratch must be ordered, with separate plans for overlapping streams.
"""
import torch

HIDDEN, INTERMEDIATE, EXPERTS, TOP_K, MAX_TOKENS = 2816, 352, 128, 8, 8


def is_available() -> bool:
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._xpu_C, "gemma_moe_fused_decode_interface")


def supports(dtype: torch.dtype, weight_dtype: torch.dtype,
             num_tokens: int, hidden_size: int, intermediate_size: int,
             num_experts: int, top_k: int) -> bool:
    return (is_available() and dtype in (torch.float16, torch.bfloat16)
            and weight_dtype == torch.float8_e4m3fn
            and 1 <= num_tokens <= MAX_TOKENS
            and (hidden_size, intermediate_size, num_experts, top_k)
            == (HIDDEN, INTERMEDIATE, EXPERTS, TOP_K))


def workspace_bytes(num_tokens: int) -> int:
    if not 1 <= num_tokens <= MAX_TOKENS:
        raise ValueError("Gemma fused decode supports 1..8 tokens")
    rows = num_tokens * TOP_K
    packed_offset = (4 * (2 + 5 * rows) + 255) // 256 * 256
    return packed_offset + 2 * rows * (2 * HIDDEN + 3 * INTERMEDIATE)


class XpuGemmaMoEFusedDecode:
    """Holds active expert tensors and per-M, device-bound uint8 workspace."""

    def __init__(self, w13: torch.Tensor, s13: torch.Tensor,
                 w2: torch.Tensor, s2: torch.Tensor):
        self.device = w13.device
        if self.device.type != "xpu":
            raise ValueError("expert weights must be on XPU")
        for name, tensor in (("w13", w13), ("s13", s13),
                             ("w2", w2), ("s2", s2)):
            if tensor.device != self.device or not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous on the weights device")
        if (w13.shape != (EXPERTS, HIDDEN, 2 * INTERMEDIATE)
                or w2.shape != (EXPERTS, INTERMEDIATE, HIDDEN)
                or s13.shape != (EXPERTS,) or s2.shape != (EXPERTS,)
                or w13.dtype != torch.float8_e4m3fn
                or w2.dtype != torch.float8_e4m3fn
                or s13.dtype != torch.float32 or s2.dtype != torch.float32):
            raise ValueError("unsupported Gemma per-expert FP8 layout")
        if w13.data_ptr() % 64 or w2.data_ptr() % 64:
            raise ValueError("expert matrices require 64-byte-aligned bases")
        self.w13, self.s13, self.w2, self.s2 = w13, s13, w2, s2
        self._workspace: dict[int, torch.Tensor] = {}

    def workspace(self, num_tokens: int) -> torch.Tensor:
        if num_tokens not in self._workspace:
            self._workspace[num_tokens] = torch.empty(
                workspace_bytes(num_tokens), dtype=torch.uint8,
                device=self.device)
        return self._workspace[num_tokens]

    def forward(self, output: torch.Tensor, hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor) -> torch.Tensor:
        if hidden_states.device != self.device:
            raise ValueError("hidden states must be on the weights device")
        if hidden_states.ndim != 2:
            raise ValueError("hidden states must be rank two")
        torch.ops._xpu_C.gemma_moe_fused_decode_interface(
            output, hidden_states, topk_weights, topk_ids,
            self.w13, self.s13, self.w2, self.s2,
            self.workspace(hidden_states.shape[0]))
        return output
