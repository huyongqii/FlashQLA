# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Blackwell entry points for Gated Delta Rule chunk kernels.

The current implementation is intentionally explicit: native sm_100 kernels are
not implemented yet, so this module forwards to the Hopper-compatible kernels
unless `FLASHQLA_REQUIRE_BLACKWELL_NATIVE=1` is set. This keeps the dispatch
boundary stable while preventing accidental claims that the compatibility path
is Blackwell-native.
"""

from __future__ import annotations

import os
import warnings

from flash_qla.ops.gated_delta_rule.chunk.hopper import (
    correct_initial_states,
    fused_gdr_bwd as _hopper_fused_gdr_bwd,
    fused_gdr_fwd as _hopper_fused_gdr_fwd,
    fused_gdr_h as _hopper_fused_gdr_h,
    get_warmup_chunks,
    kkt_solve as _hopper_kkt_solve,
)


_USE_EXPERIMENTAL_NATIVE = (
    os.environ.get("FLASHQLA_BLACKWELL_NATIVE", "") == "1"
    or os.environ.get("FLASHQLA_REQUIRE_BLACKWELL_NATIVE", "") == "1"
)
_NATIVE_KERNELS = {
    item.strip().lower()
    for item in os.environ.get("FLASHQLA_BLACKWELL_NATIVE_KERNELS", "").split(",")
    if item.strip()
}

if _USE_EXPERIMENTAL_NATIVE:
    if "fwd" in _NATIVE_KERNELS or "all" in _NATIVE_KERNELS:
        from .fused_fwd_native import fused_gdr_fwd as _native_fused_gdr_fwd
    else:
        _native_fused_gdr_fwd = None
    if "kkt" in _NATIVE_KERNELS or "all" in _NATIVE_KERNELS:
        from .kkt_solve import kkt_solve as _native_kkt_solve
    else:
        _native_kkt_solve = None
else:
    _native_fused_gdr_fwd = None
    _native_kkt_solve = None


HAS_NATIVE_BLACKWELL_KERNELS = _USE_EXPERIMENTAL_NATIVE
_WARNING_EMITTED = False
_DEBUG_EMITTED = False
_DEBUG_MESSAGES = set()


def _debug_enabled() -> bool:
    return os.environ.get("FLASHQLA_DEBUG_BLACKWELL_DISPATCH", "") == "1"


def _debug_dispatch(message: str):
    if _debug_enabled():
        if os.environ.get("FLASHQLA_DEBUG_BLACKWELL_DISPATCH_REPEAT", "") != "1":
            if message in _DEBUG_MESSAGES:
                return
            _DEBUG_MESSAGES.add(message)
        print(f"[FlashQLA Blackwell dispatch] {message}", flush=True)


def _require_or_warn(kernel_name: str):
    if os.environ.get("FLASHQLA_REQUIRE_BLACKWELL_NATIVE", "") == "1":
        raise NotImplementedError(
            f"Blackwell-native {kernel_name} is not implemented yet. "
            "Current FlashQLA sm_100 support uses the Hopper-compatible "
            "TileLang/TVM path, which has been observed to lower to HMMA rather "
            "than tcgen05/TMEM on B200. Unset "
            "FLASHQLA_REQUIRE_BLACKWELL_NATIVE to allow compatibility fallback."
        )

    global _WARNING_EMITTED
    if not _WARNING_EMITTED and os.environ.get("FLASHQLA_SUPPRESS_BLACKWELL_WARNING", "") != "1":
        _WARNING_EMITTED = True
        warnings.warn(
            "FlashQLA Blackwell native kernels are not implemented yet; "
            "falling back to the Hopper-compatible TileLang/TVM path. "
            "Set FLASHQLA_REQUIRE_BLACKWELL_NATIVE=1 to fail instead.",
            stacklevel=3,
        )


def kkt_solve(*args, **kwargs):
    if _native_kkt_solve is not None:
        if os.environ.get("FLASHQLA_BLACKWELL_KKT_EXPERIMENT", "") == "tcgen05":
            _debug_dispatch("kkt_solve=native_tcgen05_experiment")
        else:
            _debug_dispatch("kkt_solve=native_fixed_fast_candidate")
        return _native_kkt_solve(*args, **kwargs)
    if _USE_EXPERIMENTAL_NATIVE:
        _debug_dispatch("kkt_solve=hopper_fallback")
        return _hopper_kkt_solve(*args, **kwargs)
    _require_or_warn("kkt_solve")
    return _hopper_kkt_solve(*args, **kwargs)


def fused_gdr_fwd(*args, **kwargs):
    if _native_fused_gdr_fwd is not None:
        _debug_dispatch("fused_gdr_fwd=native_candidate")
        return _native_fused_gdr_fwd(*args, **kwargs)
    if _USE_EXPERIMENTAL_NATIVE:
        _debug_dispatch("fused_gdr_fwd=hopper_fallback")
        return _hopper_fused_gdr_fwd(*args, **kwargs)
    _require_or_warn("fused_gdr_fwd")
    return _hopper_fused_gdr_fwd(*args, **kwargs)


def fused_gdr_h(*args, **kwargs):
    _require_or_warn("fused_gdr_h")
    return _hopper_fused_gdr_h(*args, **kwargs)


def fused_gdr_bwd(*args, **kwargs):
    _require_or_warn("fused_gdr_bwd")
    return _hopper_fused_gdr_bwd(*args, **kwargs)


__all__ = [
    "HAS_NATIVE_BLACKWELL_KERNELS",
    "fused_gdr_fwd",
    "fused_gdr_bwd",
    "fused_gdr_h",
    "kkt_solve",
    "get_warmup_chunks",
    "correct_initial_states",
]
