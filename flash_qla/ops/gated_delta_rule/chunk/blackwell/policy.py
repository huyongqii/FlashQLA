# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Runtime policy for experimental Blackwell GDR kernels."""

from __future__ import annotations

import os


def blackwell_fwd_policy() -> str:
    policy = os.environ.get("FLASHQLA_BLACKWELL_FWD_POLICY", "native").strip().lower()
    if policy in ("", "auto"):
        return "auto"
    if policy in ("native", "force_native"):
        return "native"
    if policy in ("compat", "fallback", "hopper"):
        raise ValueError(
            "Blackwell Hopper-compatible fallback has been removed. "
            "Use FLASHQLA_BLACKWELL_FWD_POLICY=native or auto."
        )
    raise ValueError(
        "FLASHQLA_BLACKWELL_FWD_POLICY must be one of auto or native, "
        f"got {policy!r}"
    )


def should_use_native_fwd(num_v_heads: int, num_k_heads: int) -> tuple[bool, str]:
    """Return whether the current Blackwell native fwd path should be used.

    The default policy is native when `FLASHQLA_BLACKWELL_NATIVE=1` selects
    this module. `auto` remains available as an explicit hybrid policy for
    production guardrails while the Qwen TP2/TP4/TP8 small-V-head kernels are
    being specialized.
    """

    del num_k_heads
    policy = blackwell_fwd_policy()
    if policy == "native":
        return True, "forced_native"
    if num_v_heads >= 64:
        return True, "auto_hv64"
    return False, f"auto_small_hv{num_v_heads}"
