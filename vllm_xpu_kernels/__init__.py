# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec

from .flash_attn_interface import flash_attn_varlen_func  # noqa: F401

# AOT artifacts can bypass pass initialization; register their optional ops
# when loading the package rather than only from the compile pass.
GEMMA_COMPILE_KERNELS_AVAILABLE = (
    find_spec(f"{__name__}._gemma_compile_C") is not None
)
if GEMMA_COMPILE_KERNELS_AVAILABLE:
    from . import _gemma_compile_C  # noqa: F401
