# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang


def profile(func, inputs, wait: int = 50, warmup: int = 50, rep: int = 100):
    """Profile `func(*inputs)` and return a dict {kernel_name: ms}.

    Notes on key naming:
      * `FunctionEventAvg.key` in newer PyTorch versions can be the *aten op*
        name rather than the underlying CUDA / Triton kernel name (e.g.
        "aten::matmul" instead of "cutlass::Kernel2<...>"). Using the raw key
        therefore breaks downstream lookups by kernel name.
      * We instead use the human-readable event name `event.key` joined with
        the actual GPU device time. For pure CUDA / Triton kernels this is
        identical to what `nsys stats` reports.
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=rep),
    ) as prof:
        for _ in range(wait + warmup + rep):
            func(*inputs)
            prof.step()

    result: dict[str, float] = {}
    for x in prof.key_averages():
        # Prefer the CUDA-side timing field. PyTorch renamed
        # `cuda_time_total` -> `device_time_total` around 2.4; support both.
        for attr in ("device_time_total", "cuda_time_total", "device_time"):
            t = getattr(x, attr, None)
            if t:
                break
        else:
            t = 0.0
        if t <= 0:
            # CPU-only event; skip so the dict contains only GPU work.
            continue
        # `x.key` is normally the aten op name. For kernels launched directly
        # (Triton / TileLang / cutlass) torch.profiler stores the kernel name
        # under a child event. We look it up via `.kernel_backend` / event
        # list; if unavailable, fall back to `x.key`.
        name = getattr(x, "key", None) or "<unknown>"
        # Accumulate (some kernels appear under multiple aten parents).
        result[name] = result.get(name, 0.0) + t * 1e-3  # us -> ms

    result["total"] = tilelang.profiler.do_bench(
        lambda: func(*inputs), warmup=warmup, rep=rep
    )
    return result
