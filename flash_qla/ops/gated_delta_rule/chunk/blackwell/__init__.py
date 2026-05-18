# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Blackwell entry points for Gated Delta Rule chunk kernels."""

from __future__ import annotations

from .cp_fwd import correct_initial_states, get_warmup_chunks
from .fused_fwd_native import fused_gdr_fwd
from .kkt_solve import kkt_solve
from .prepare_h import fused_gdr_h

HAS_NATIVE_BLACKWELL_KERNELS = True


def fused_gdr_bwd(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("Blackwell fused_gdr_bwd is not implemented.")


__all__ = [
    "HAS_NATIVE_BLACKWELL_KERNELS",
    "fused_gdr_fwd",
    "fused_gdr_bwd",
    "fused_gdr_h",
    "kkt_solve",
    "get_warmup_chunks",
    "correct_initial_states",
]
