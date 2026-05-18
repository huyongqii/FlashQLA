# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Blackwell entry points for Gated Delta Rule chunk kernels.

Only native Blackwell kernels are dispatched from this module. Unsupported
features fail explicitly instead of falling back to Hopper-compatible kernels;
this keeps B200/B300 debugging focused on one implementation path.
"""

from __future__ import annotations

import os


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

        _NATIVE_FWD_NAME = "ag"
    else:
        _native_fused_gdr_fwd = None
        _NATIVE_FWD_NAME = "none"
    if "kkt" in _NATIVE_KERNELS or "all" in _NATIVE_KERNELS:
        from .kkt_solve import kkt_solve as _native_kkt_solve
    else:
        _native_kkt_solve = None
    from .prepare_h import fused_gdr_h as _native_fused_gdr_h
else:
    _native_fused_gdr_fwd = None
    _native_kkt_solve = None
    _native_fused_gdr_h = None
    _NATIVE_FWD_NAME = "none"


HAS_NATIVE_BLACKWELL_KERNELS = _USE_EXPERIMENTAL_NATIVE
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


def _unsupported(kernel_name: str):
    raise NotImplementedError(
        f"Blackwell-native {kernel_name} is not available in the current build. "
        "FlashQLA no longer falls back to Hopper-compatible kernels on Blackwell. "
        "Enable the native kernel in FLASHQLA_BLACKWELL_NATIVE_KERNELS or use a "
        "feature path that has a Blackwell implementation."
    )


def kkt_solve(*args, **kwargs):
    if _native_kkt_solve is not None:
        if os.environ.get("FLASHQLA_BLACKWELL_KKT_EXPERIMENT", "") == "tcgen05":
            _debug_dispatch("kkt_solve=native_tcgen05_experiment")
        else:
            _debug_dispatch("kkt_solve=native_fixed_fast_candidate")
        return _native_kkt_solve(*args, **kwargs)
    _unsupported("kkt_solve")


def fused_gdr_fwd(*args, **kwargs):
    if _native_fused_gdr_fwd is not None:
        _debug_dispatch(f"fused_gdr_fwd=native_{_NATIVE_FWD_NAME}")
        return _native_fused_gdr_fwd(*args, **kwargs)
    _unsupported("fused_gdr_fwd")


def fused_gdr_h(*args, **kwargs):
    if _native_fused_gdr_h is not None:
        _debug_dispatch("fused_gdr_h=native_prepare_h")
        return _native_fused_gdr_h(*args, **kwargs)
    _unsupported("fused_gdr_h")


def fused_gdr_bwd(*args, **kwargs):
    _unsupported("fused_gdr_bwd")


def get_warmup_chunks(*args, **kwargs):
    _unsupported("get_warmup_chunks")


def correct_initial_states(*args, **kwargs):
    _unsupported("correct_initial_states")


__all__ = [
    "HAS_NATIVE_BLACKWELL_KERNELS",
    "fused_gdr_fwd",
    "fused_gdr_bwd",
    "fused_gdr_h",
    "kkt_solve",
    "get_warmup_chunks",
    "correct_initial_states",
]
