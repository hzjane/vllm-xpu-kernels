# SPDX-License-Identifier: Apache-2.0
"""Capability decisions must fail closed without requiring XPU allocations."""

from types import SimpleNamespace

import pytest
import torch

from vllm_xpu_kernels import gemma_moe_fused_interface as interface


@pytest.mark.parametrize(
    "xe2,xe3,expected",
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_availability_rejects_unsupported_architecture(
    monkeypatch, xe2, xe3, expected
):
    ops = SimpleNamespace(
        gemma_moe_fused_decode_interface=object(),
        gemma_moe_topk_interface=object(),
        is_xe2_arch=lambda *args: xe2,
        is_xe3_arch=lambda *args: xe3,
    )
    monkeypatch.setattr(torch.ops, "_xpu_C", ops)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    assert interface.is_available() is expected


def test_availability_rejects_registered_schemas_without_xpu(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert not interface.is_available()


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("xpu:7")])
def test_availability_checks_explicit_target_device(monkeypatch, device):
    seen = []

    def xe2(index):
        seen.append(index)
        return index == 7

    ops = SimpleNamespace(
        gemma_moe_fused_decode_interface=object(),
        gemma_moe_topk_interface=object(),
        is_xe2_arch=xe2,
        is_xe3_arch=lambda *args: False,
    )
    monkeypatch.setattr(torch.ops, "_xpu_C", ops)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    result = interface.supports(
        torch.float16, torch.float8_e4m3fn, 1, 2816, 352, 128, 8, device=device
    )
    assert result is (device.type == "xpu")
    assert seen == ([7] if device.type == "xpu" else [])


def test_availability_rejects_build_without_gemma_registration(monkeypatch):
    monkeypatch.setattr(
        torch.ops,
        "_xpu_C",
        SimpleNamespace(
            is_xe2_arch=lambda *args: True, is_xe3_arch=lambda *args: False
        ),
    )
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    assert not interface.is_available()
