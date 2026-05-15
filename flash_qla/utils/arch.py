# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""
Arch detection utility for FlashQLA.

Supported compute capabilities:
    - "9.0"    : Hopper (H100 / H200)         -> uses flash_qla.ops.gated_delta_rule.chunk.hopper
    - "10.x"   : Blackwell datacenter (B200/B300) -> Blackwell dispatch path
    - "10.xa"  : Blackwell accelerated targets -> same as above

Override via environment variable:
    FLASHQLA_FORCE_ARCH=sm90    # force hopper path
    FLASHQLA_FORCE_ARCH=sm100   # force blackwell path
    FLASHQLA_FORCE_ARCH=sm103   # force blackwell path for B300-like targets
"""

import os
import warnings

import tilelang


# Compute capabilities for which we currently dispatch to the hopper-style
# (4-warpgroup, warp-specialized, WGMMA / TMA) implementation in
# flash_qla.ops.gated_delta_rule.chunk.hopper.
#
# Blackwell devices can report different minor compute capabilities, e.g.
# B200 has been observed as sm_100 while B300 reports sm_103. Treat all 10.x
# CUDA targets as Blackwell-like for dispatch, then let the TileLang/CUDA
# backend decide the exact codegen target.
_HOPPER_LIKE_CCS = {"9.0"}


_FORCE_ARCH_TO_CC = {
    "sm90": "9.0",
    "sm_90": "9.0",
    "9.0": "9.0",
    "hopper": "9.0",
    "sm100": "10.0",
    "sm_100": "10.0",
    "sm100a": "10.0a",
    "sm_100a": "10.0a",
    "sm103": "10.3",
    "sm_103": "10.3",
    "sm103a": "10.3a",
    "sm_103a": "10.3a",
    "10.0": "10.0",
    "10.0a": "10.0a",
    "10.3": "10.3",
    "10.3a": "10.3a",
    "blackwell": "10.0",
}


def get_compute_capability() -> str:
    """Return the compute capability string, e.g. "9.0" or "10.0".

    Honors FLASHQLA_FORCE_ARCH for testing / fallback.
    """
    force = os.environ.get("FLASHQLA_FORCE_ARCH", "").strip().lower()
    if force:
        if force not in _FORCE_ARCH_TO_CC:
            raise ValueError(
                f"FLASHQLA_FORCE_ARCH={force!r} is not recognized. "
                f"Allowed: {sorted(_FORCE_ARCH_TO_CC.keys())}"
            )
        return _FORCE_ARCH_TO_CC[force]
    return tilelang.contrib.nvcc.get_target_compute_version()


def is_hopper(cc: str | None = None) -> bool:
    cc = cc or get_compute_capability()
    return cc in _HOPPER_LIKE_CCS


def is_blackwell(cc: str | None = None) -> bool:
    cc = cc or get_compute_capability()
    return cc.startswith("10.")


def _cc_to_sm(cc: str) -> str:
    return f"sm_{cc.replace('.', '')}"


_BLACKWELL_WARNING_EMITTED = False


def assert_supported(cc: str | None = None) -> str:
    global _BLACKWELL_WARNING_EMITTED

    cc = cc or get_compute_capability()
    if cc not in _HOPPER_LIKE_CCS and not is_blackwell(cc):
        raise ValueError(
            f"FlashQLA does not support compute capability {_cc_to_sm(cc)}. "
            f"Supported: sm_90 (Hopper), sm_10x / sm_10xa (Blackwell). "
            f"Set FLASHQLA_FORCE_ARCH=sm90|sm100|sm103 to override."
        )
    if is_blackwell(cc) and not _BLACKWELL_WARNING_EMITTED:
        # Tier-1: reuse hopper kernels on Blackwell. Warn ONCE per process so
        # users know perf may be sub-optimal until Tier-3 specialized kernels
        # are in place.
        _BLACKWELL_WARNING_EMITTED = True
        if os.environ.get("FLASHQLA_SUPPRESS_BLACKWELL_WARNING", "") != "1":
            warnings.warn(
                f"FlashQLA is running on {_cc_to_sm(cc)} (Blackwell). "
                "B200/B300 support is experimental; enable the native "
                "Blackwell path for tcgen05/TMEM kernels and verify generated "
                "SASS on the target GPU. "
                "Set FLASHQLA_SUPPRESS_BLACKWELL_WARNING=1 to silence.",
                stacklevel=2,
            )
    return cc
