# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec

from .flash_attn_interface import flash_attn_varlen_func  # noqa: F401

RMS_KERNELS_AVAILABLE = find_spec(f"{__name__}._rms_C") is not None
if RMS_KERNELS_AVAILABLE:
    from . import _rms_C  # noqa: F401

# Optional module: missing permits fallback, present ABI errors must propagate.
DECODE_AUX_KERNELS_AVAILABLE = (
    find_spec(f"{__name__}._decode_aux_C") is not None
)
if DECODE_AUX_KERNELS_AVAILABLE:
    from . import _decode_aux_C  # noqa: F401

MOE_DECODE_KERNELS_AVAILABLE = (
    find_spec(f"{__name__}._moe_decode_C") is not None
)
if MOE_DECODE_KERNELS_AVAILABLE:
    from . import _moe_decode_C  # noqa: F401

# The optional FP16 module specializes only small-M Router projection.
FP16_LINEAR_KERNELS_AVAILABLE = find_spec(f"{__name__}._fp16_C") is not None
if FP16_LINEAR_KERNELS_AVAILABLE:
    from . import _fp16_C  # noqa: F401
