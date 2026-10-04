# SPDX-License-Identifier: Apache-2.0
"""CPU reproductions and source guards, never a SYCL execution test."""

import ctypes
import math
from pathlib import Path

import pytest
import torch


SOURCE = (Path(__file__).resolve().parents[1] /
          "csrc/qwen38/gdn_sycl.cpp").read_text()


@pytest.mark.parametrize("padded,separate_storage", [
    (True, False), (True, True), (False, True)
])
def test_cpu_physical_alias_can_escape_torch_overlap_assert(
        padded, separate_storage):
    # Inspect the existing CPU Torch implementation, without compiling code.
    library = ctypes.CDLL(str(Path(torch.__file__).parent /
                              "lib/libtorch_cpu.so"))
    overlap = getattr(
        library, "_ZN2at18get_overlap_statusEPKN3c1010TensorImplES3_", None)
    if overlap is None:
        pytest.skip("Torch does not export the TensorImpl overlap probe")
    overlap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    overlap.restype = ctypes.c_int
    buffer = bytearray(64)
    base = torch.frombuffer(buffer, dtype=torch.float16).view(2, 16)
    other = (torch.frombuffer(buffer, dtype=torch.float16).view(2, 16)
             if separate_storage else base)
    reader = base[:, :8] if padded else base
    aliased = other.flatten()[:16].view(2, 8)
    assert reader.data_ptr() == aliased.data_ptr()
    assert torch._C._overlaps(reader, aliased) is not separate_storage
    # Full=0, Partial=1, No=2, TooHard=3 in ATen/MemoryOverlap.h.
    assert overlap(reader._cdata, aliased._cdata) == (3 if padded else 2)
    assert "MemOverlapStatus::TooHard" in SOURCE
    assert "storage().nbytes()" in SOURCE
    checks = SOURCE.split("void check_no_cross_alias(", 1)[1]
    checks = checks.split("\n}\n", 1)[0]
    assert "check_no_overlap(*a, *b);" in checks
    # Dense separate Storage owners need a pointer check even for status No.
    helper = SOURCE.split("void check_no_overlap(", 1)[1]
    helper = helper.split("\n}\n", 1)[0]
    assert "a.const_data_ptr()" in helper
    assert "b.const_data_ptr()" in helper


@pytest.mark.parametrize("name,first_submit", [
    ("gdn_decode_sycl", "launch_decode_root("),
    ("gdn_spec_v2_sycl", "launch_conv<true>("),
    ("gdn_norm_gate_sycl", "queue.parallel_for<"),
    ("gdn_spec_conv_probe_sycl", "launch_conv<true>("),
])
def test_cpu_gdn_records_ownership_before_first_submit(name, first_submit):
    body = SOURCE.split(f"void {name}(", 1)[1].split("\n}\n", 1)[0]
    assert body.index("record_stream(") < body.index(first_submit)
    if name in ("gdn_decode_sycl", "gdn_spec_v2_sycl"):
        allocation = body.index("auto qkv = at::empty(")
        assert allocation < body.index("record_stream({&qkv}, stream);")
        assert body.index("record_stream({&qkv}, stream);") < body.index(
            "launch_conv<")


@pytest.mark.parametrize("scale", [1e300, 1e-300])
def test_cpu_gdn_scale_must_remain_positive_finite_fp32(scale):
    assert math.isfinite(scale) and scale > 0
    converted = torch.tensor(scale, dtype=torch.float64).float().item()
    assert not (math.isfinite(converted) and converted > 0)
    assert "std::isfinite(scale_fp32) && scale_fp32 > 0.0f" in SOURCE


def test_cpu_gdn_accepted_subtraction_cannot_overflow_int32():
    accepted = torch.tensor([-2**31], dtype=torch.int32)
    assert int((accepted - 1)[0]) == 2**31 - 1  # Previous int32 expression.
    assert int((accepted.long() - 1)[0]) == -2**31 - 1
    lines = [line for line in SOURCE.splitlines()
             if "accepted[" in line and "- 1" in line]
    assert len(lines) == 4
    assert all("int64_t(accepted[" in line for line in lines)


def test_cpu_gdn_projection_checks_stream_before_fused_or_two_stage_path():
    source = (Path(__file__).resolve().parents[1] /
              "csrc/qwen38/gdn_sycl_projection.cpp").read_text()
    body = source.split("void gdn_norm_int4_sycl(", 1)[1]
    # External SYCL queues can be unordered. The two-stage path otherwise
    # allows projection to read normalized before norm_gate has written it.
    assert body.index("queue.is_in_order()") < body.index("if (hv == kHeads")
