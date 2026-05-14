# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""
Arch detection utility for FlashQLA.

Supported compute capabilities:
    - "9.0"    : Hopper (H100 / H200)         -> uses flash_qla.ops.gated_delta_rule.chunk.hopper
    - "10.0"   : Blackwell datacenter (B200/B300) -> reuse hopper kernels for now (Tier-1 minimal support)
    - "10.0a"  : Blackwell sm_100a            -> same as above

Override via environment variable:
    FLASHQLA_FORCE_ARCH=sm90    # force hopper path
    FLASHQLA_FORCE_ARCH=sm100   # force blackwell path
"""

import os
import warnings

import tilelang


# Compute capabilities for which we currently dispatch to the hopper-style
# (4-warpgroup, warp-specialized, WGMMA / TMA) implementation in
# flash_qla.ops.gated_delta_rule.chunk.hopper.
#
# For Blackwell (sm_100 / sm_100a) we reuse the hopper kernels at Tier-1; the
# TileLang backend is expected to lower T.gemm to tcgen05.mma automatically
# when targeting sm_100a. A dedicated `blackwell/` directory may be added later
# for Tier-3 specialization.
_HOPPER_LIKE_CCS = {"9.0"}
_BLACKWELL_LIKE_CCS = {"10.0", "10.0a", "11.0"}
_SUPPORTED_CCS = _HOPPER_LIKE_CCS | _BLACKWELL_LIKE_CCS


_FORCE_ARCH_TO_CC = {
    "sm90": "9.0",
    "sm_90": "9.0",
    "9.0": "9.0",
    "hopper": "9.0",
    "sm100": "10.0",
    "sm_100": "10.0",
    "sm100a": "10.0a",
    "sm_100a": "10.0a",
    "10.0": "10.0",
    "10.0a": "10.0a",
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
    return cc in _BLACKWELL_LIKE_CCS


_BLACKWELL_WARNING_EMITTED = False


def assert_supported(cc: str | None = None) -> str:
    global _BLACKWELL_WARNING_EMITTED

    cc = cc or get_compute_capability()
    if cc not in _SUPPORTED_CCS:
        raise ValueError(
            f"FlashQLA does not support compute capability sm_{cc.replace('.', '')}. "
            f"Supported: sm_90 (Hopper), sm_100 / sm_100a (Blackwell). "
            f"Set FLASHQLA_FORCE_ARCH=sm90|sm100 to override."
        )
    if cc in _BLACKWELL_LIKE_CCS and not _BLACKWELL_WARNING_EMITTED:
        # Tier-1: reuse hopper kernels on Blackwell. Warn ONCE per process so
        # users know perf may be sub-optimal until Tier-3 specialized kernels
        # are in place.
        _BLACKWELL_WARNING_EMITTED = True
        if os.environ.get("FLASHQLA_SUPPRESS_BLACKWELL_WARNING", "") != "1":
            warnings.warn(
                "FlashQLA is running on sm_100 (Blackwell) using the "
                "Hopper-compatible kernel path. Current TileLang/TVM lowering "
                "has been observed to emit HMMA rather than tcgen05/TMEM on "
                "B200, so performance is not yet tuned for B200/B300. "
                "Set FLASHQLA_SUPPRESS_BLACKWELL_WARNING=1 to silence.",
                stacklevel=2,
            )
    return cc
