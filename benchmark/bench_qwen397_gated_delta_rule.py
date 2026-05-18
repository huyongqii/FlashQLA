#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
#
# Blackwell-side benchmark aligned with benchmark/bench_gated_delta_rule.py.

import argparse
import gc
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import tilelang


HEAD_DIM = 128


@dataclass
class ModelConfig:
    label: str
    h_qk: int
    h_v: int


@dataclass
class SeqLenConfig:
    label: str
    seqlens: List[int]


FWD_MODEL_CONFIGS = [
    ModelConfig("397B/122B TP4", 4, 16),
    ModelConfig("397B/122B TP2", 8, 32),
]

FWD_SEQLEN_CONFIGS = [
    SeqLenConfig("1x32768", [32768]),
    SeqLenConfig("1x16384", [16384]),
    SeqLenConfig("1x8192", [8192]),
    SeqLenConfig("1x4096", [4096]),
    SeqLenConfig("1x2048", [2048]),
    SeqLenConfig("28672+4096", [28672, 4096]),
    SeqLenConfig("24576+8192", [24576, 8192]),
    SeqLenConfig("16384+16384", [16384, 16384]),
    SeqLenConfig("8192+24576", [8192, 24576]),
    SeqLenConfig("4096+28672", [4096, 28672]),
    SeqLenConfig("12288+4096", [12288, 4096]),
    SeqLenConfig("6144+2048", [6144, 2048]),
    SeqLenConfig("4096+4096", [4096, 4096]),
    SeqLenConfig("2048+6144", [2048, 6144]),
    SeqLenConfig("1024+7168", [1024, 7168]),
    SeqLenConfig("8192x4", [8192] * 4),
    SeqLenConfig("4096x8", [4096] * 8),
    SeqLenConfig("2048x4", [2048] * 4),
    SeqLenConfig("1024x8", [1024] * 8),
]

def parse_env_overrides(items: List[str]) -> Dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env has an empty key in {item!r}")
        result[key] = value
    return result


def apply_env_overrides(extra_env: Dict[str, str]) -> Dict[str, str]:
    os.environ.update(extra_env)
    os.environ.setdefault("FLASHQLA_SUPPRESS_BLACKWELL_WARNING", "1")
    return {key: os.environ[key] for key in sorted(extra_env)}


def cleanup_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_lib_versions() -> Dict[str, str]:
    versions = {}
    try:
        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = "N/A"
    try:
        import fla

        versions["fla"] = getattr(fla, "__version__", "Installed (ver unknown)")
    except ImportError:
        versions["fla"] = "Not Installed"
    try:
        versions["tilelang"] = getattr(tilelang, "__version__", "Installed (ver unknown)")
    except Exception:
        versions["tilelang"] = "N/A"
    return versions


def prepare_tensors(
    seqlens: List[int], h_qk: int, h_v: int, l2norm, head_dim: int = HEAD_DIM
) -> Optional[Dict[str, Any]]:
    device = "cuda"
    num_seqs = len(seqlens)
    total_tokens = sum(seqlens)
    scale = head_dim ** (-0.5)

    offsets = [0]
    for s in seqlens:
        offsets.append(offsets[-1] + s)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)

    try:
        q = l2norm(
            torch.randn(
                1, total_tokens, h_qk, head_dim, device=device, dtype=torch.bfloat16
            )
        )
        k = l2norm(
            torch.randn(
                1, total_tokens, h_qk, head_dim, device=device, dtype=torch.bfloat16
            )
        )
        v = torch.randn(
            1, total_tokens, h_v, head_dim, device=device, dtype=torch.bfloat16
        )
        g = (
            F.logsigmoid(
                torch.randn(1, total_tokens, h_v, device=device, dtype=torch.float32)
            )
            / 16
        )
        beta = torch.randn(
            1, total_tokens, h_v, device=device, dtype=torch.float32
        ).sigmoid()
        h0 = torch.randn(
            num_seqs, h_v, head_dim, head_dim, device=device, dtype=torch.float32
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return None
        raise e

    swa_ratio = 0.75
    swa_mask = torch.zeros(h_v, dtype=torch.bool, device=device)
    swa_mask[: math.ceil(swa_ratio * h_v)] = True
    swa_mask = swa_mask[torch.randperm(h_v, device=device)]
    g[:, :, ~swa_mask] = 0.0

    return {
        "scale": scale,
        "cu_seqlens": cu_seqlens,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "h0": h0,
    }


def bench_fwd(
    seqlens: List[int],
    h_qk: int,
    h_v: int,
    qla_fwd,
    fla_fwd,
    l2norm,
    head_dim: int = HEAD_DIM,
    warmup: int = 10,
    repeats: int = 5,
    auto_cp: bool = True,
) -> Tuple[float, float]:
    cleanup_cuda()
    data = prepare_tensors(seqlens, h_qk, h_v, l2norm, head_dim)
    if data is None:
        return float("nan"), float("nan")

    q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
    h0, scale, cu_seqlens = data["h0"], data["scale"], data["cu_seqlens"]

    def call_qla_fwd():
        qla_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            output_h=False,
            cu_seqlens=cu_seqlens,
            auto_cp=auto_cp,
        )

    def call_fla_fwd():
        fla_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )

    try:
        qla_ms = tilelang.profiler.do_bench(call_qla_fwd, warmup=warmup, rep=repeats)
    except RuntimeError as e:
        print(f"\n[WARN] FlashQLA Fwd failed: {e}")
        cleanup_cuda()
        qla_ms = float("nan")

    try:
        fla_ms = tilelang.profiler.do_bench(call_fla_fwd, warmup=warmup, rep=repeats)
    except RuntimeError as e:
        print(f"\n[WARN] FLA Fwd failed: {e}")
        cleanup_cuda()
        fla_ms = float("nan")

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    return qla_ms, fla_ms


FWD_HDR = (
    f"{'Model Config':<16} {'Seqlens':<17} {'h_qk':>5} {'h_v':>5}    "
    f"{'flash_qla [fwd]':>10}  {'FLA [fwd]':>10}   {'vs FLA':>7}"
)


def fmt_time(ms: float) -> str:
    if math.isnan(ms):
        return "     N/A  "
    return f"{ms:>8.3f}ms"


def fmt_ratio(base: float, other: float) -> str:
    if math.isnan(base) or math.isnan(other) or base == 0:
        return "   N/A  "
    return f"{other / base:>6.2f}x"


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen397 FlashQLA GDR Fwd")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env override, e.g. --env FLASHQLA_CP_MIN_CHUNKS=1.",
    )
    args = parser.parse_args()

    visible_env = apply_env_overrides(parse_env_overrides(args.env))

    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_fwd
    from flash_qla import chunk_gated_delta_rule_fwd as qla_fwd
    from flash_qla.utils import l2norm

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"GPU: {gpu_name}")
    print("Models: Qwen3.5 397B/122B TP2/TP4, d=128")
    if visible_env:
        print(f"Env: {visible_env}")
    print(f"Config: Warmup={args.warmup}, Repeats={args.repeats}")

    libs = get_lib_versions()
    print("Library Versions:")
    ver_str = " | ".join([f"{k}: {v}" for k, v in libs.items()])
    print(f"  {ver_str}")

    print("=" * 90)
    print("\n>>> FORWARD BENCHMARKS")
    print(FWD_HDR)
    print("-" * len(FWD_HDR))

    prev_model = None
    for cfg in FWD_MODEL_CONFIGS:
        if prev_model is not None and cfg.label != prev_model:
            print()
        prev_model = cfg.label

        for sl_cfg in FWD_SEQLEN_CONFIGS:
            try:
                qla_ms, fla_ms = bench_fwd(
                    sl_cfg.seqlens,
                    cfg.h_qk,
                    cfg.h_v,
                    qla_fwd,
                    fla_fwd,
                    l2norm,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    auto_cp=True,
                )

                ratio_fla = fmt_ratio(qla_ms, fla_ms)

                print(
                    f"{cfg.label:<16} {sl_cfg.label:<17} {cfg.h_qk:>5} {cfg.h_v:>5}    "
                    f"{fmt_time(qla_ms)}  {fmt_time(fla_ms)}   {ratio_fla}",
                    flush=True,
                )
            except Exception as e:
                print(f"\n[ERROR] Forward Case Failed: {cfg.label} / {sl_cfg.label}")
                print(f"Exception: {e}")
                cleanup_cuda()
                continue

            cleanup_cuda()

    print("\nBenchmark Finished.")


if __name__ == "__main__":
    main()
