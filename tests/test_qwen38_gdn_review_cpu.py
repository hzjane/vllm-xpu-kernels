# SPDX-License-Identifier: Apache-2.0
"""CPU reproductions and source guards, never a SYCL execution test."""

import ctypes
import math
import random
from pathlib import Path

import pytest
import torch


SOURCE = (Path(__file__).resolve().parents[1] /
          "csrc/qwen38/gdn_sycl.cpp").read_text()


def test_cpu_pitched_overlap_algorithm_matches_explicit_bytes():
    def optimized(a, b):
        ab, aw, ap, ar = a
        bb, bw, bp, br = b
        if not ar or not br or ab + (ar - 1) * ap + aw <= bb or \
                bb + (br - 1) * bp + bw <= ab:
            return False
        if ap == bp:
            if ab > bb:
                ab, aw, ap, ar, bb, bw, bp, br = bb, bw, bp, br, ab, aw, ap, ar
            row, offset = divmod(bb - ab, ap)
            return (row < ar and offset < aw) or \
                (row + 1 < ar and bw > ap - offset)
        i = j = 0
        while i < ar and j < br:
            x, y = ab + i * ap, bb + j * bp
            if x <= y and y - x >= aw:
                i += (y - x - aw) // ap + 1
            elif y <= x and x - y >= bw:
                j += (x - y - bw) // bp + 1
            else:
                return True
        return False

    rng = random.Random(7021)
    for _ in range(10000):
        ap, bp = rng.randint(1, 32), rng.randint(1, 32)
        if rng.random() < 0.5:
            bp = ap
        a = (rng.randint(0, 200), rng.randint(1, ap), ap, rng.randint(0, 9))
        b = (rng.randint(0, 200), rng.randint(1, bp), bp, rng.randint(0, 9))
        exact = lambda t: {t[0] + row * t[2] + byte
                           for row in range(t[3]) for byte in range(t[1])}
        assert optimized(a, b) == bool(exact(a) & exact(b)), (a, b)


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
    assert "physical_overlap(a, b)" in SOURCE
    assert "GDN alias check requires dense inner rows" in SOURCE
    assert "left.pitch == right.pitch" in SOURCE
    checks = SOURCE.split("void check_no_cross_alias(", 1)[1]
    checks = checks.split("\n}\n", 1)[0]
    assert "check_no_overlap(*a, *b);" in checks
    # Dense separate Storage owners need a pointer check even for status No.
    helper = SOURCE.split("void check_no_overlap(", 1)[1]
    helper = helper.split("\n}\n", 1)[0]
    assert "physical_overlap(a, b)" in helper
    rows = SOURCE.split("MemoryRows memory_rows(", 1)[1].split("\n}\n", 1)[0]
    assert "t.const_data_ptr()" in rows


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
