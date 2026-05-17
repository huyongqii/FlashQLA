#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Compare Blackwell small-Hv fwd variants on identical inputs.

This is a focused correctness locator, not a benchmark.  It compares native,
small_hv recompute-P, and small_hv precomputed-Pg outputs and reports where the
first large differences appear by tensor index, head, and chunk.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))


@contextlib.contextmanager
def patched_env(values: dict[str, str | None]):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)


def max_report(name: str, actual: torch.Tensor, expected: torch.Tensor, chunk_size: int):
    diff = (actual.float() - expected.float()).abs()
    max_val = diff.max()
    denom = expected.float().abs().max().clamp_min(1e-6)
    flat = int(diff.argmax().item())
    idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat, device=diff.device), diff.shape))
    print(
        f"{name}: max_abs={max_val.item():.6f} rel={(max_val / denom).item():.6f} "
        f"idx={idx} actual={actual[idx].float().item():.6f} expected={expected[idx].float().item():.6f}"
    )
    if diff.ndim >= 4:
        # Expected O shape: [B, T, H, D]
        per_head = diff.amax(dim=(0, 1, 3))
        top = torch.topk(per_head, k=min(8, per_head.numel()))
        print(
            f"{name}: top_heads="
            + ", ".join(
                f"h{int(h)}:{float(v):.6f}" for v, h in zip(top.values, top.indices)
            )
        )
        num_chunks = diff.shape[1] // chunk_size
        if num_chunks > 0:
            chunk_diff = diff[:, : num_chunks * chunk_size].reshape(
                diff.shape[0], num_chunks, chunk_size, diff.shape[2], diff.shape[3]
            )
            per_chunk = chunk_diff.amax(dim=(0, 2, 3, 4))
            topc = torch.topk(per_chunk, k=min(8, per_chunk.numel()))
            print(
                f"{name}: top_chunks="
                + ", ".join(
                    f"c{int(c)}:{float(v):.6f}" for v, c in zip(topc.values, topc.indices)
                )
            )
    elif diff.ndim == 4:
        per_head = diff.amax(dim=(0, 2, 3))
        top = torch.topk(per_head, k=min(8, per_head.numel()))
        print(
            f"{name}: top_state_heads="
            + ", ".join(
                f"h{int(h)}:{float(v):.6f}" for v, h in zip(top.values, top.indices)
            )
        )


def run_qla(label: str, q, k, v, g, beta, scale, h0, env: dict[str, str | None]):
    from flash_qla import chunk_gated_delta_rule_fwd

    with patched_env(env):
        torch.cuda.synchronize()
        g_out, a_out, o, h, s = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=h0,
            cu_seqlens=None,
            output_final_state=True,
            output_h=False,
            auto_cp=False,
        )
        torch.cuda.synchronize()
    print(f"{label}: g={tuple(g_out.shape)} A={tuple(a_out.shape)} o={tuple(o.shape)} s={tuple(s.shape)}")
    return g_out, a_out, o, s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--nkh", type=int, default=2)
    parser.add_argument("--nvh", type=int, default=8)
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swa-ratio", type=float, default=0.75)
    parser.add_argument("--pg-fp32", action="store_true")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the QLA variants on the same inputs to catch intermittent failures.",
    )
    parser.add_argument(
        "--match-test-gdr-rng",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Consume the same extra random tensors as tests/test_gdr.py before "
            "building the SWA mask, so seed=42 reproduces the benchmark inputs."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE", "1")
    os.environ.setdefault("FLASHQLA_BLACKWELL_NATIVE", "1")
    os.environ.setdefault("FLASHQLA_BLACKWELL_NATIVE_KERNELS", "fwd,kkt")
    os.environ.setdefault("FLASHQLA_BLACKWELL_FWD_POLICY", "native")
    os.environ.setdefault("FLASHQLA_BLACKWELL_BLOCK_DV", "64")
    os.environ.setdefault("FLASHQLA_BLACKWELL_FWD_THREADS", "256")
    os.environ.setdefault("FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS", "load,h")

    from ref_gdr import chunk_gated_delta_rule_fwd as ref_fwd

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = l2norm(torch.randn(args.batch, args.tokens, args.nkh, args.dk, device=device, dtype=dtype))
    k = l2norm(torch.randn(args.batch, args.tokens, args.nkh, args.dk, device=device, dtype=dtype))
    v = torch.randn(args.batch, args.tokens, args.nvh, args.dv, device=device, dtype=dtype)
    g = torch.nn.functional.logsigmoid(
        torch.randn(args.batch, args.tokens, args.nvh, device=device, dtype=torch.float32)
    ) / 16
    beta = torch.randn(args.batch, args.tokens, args.nvh, device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn(args.batch, args.nvh, args.dk, args.dv, device=device, dtype=torch.float32)
    if args.match_test_gdr_rng:
        _do = torch.randn_like(v)
        _dht = torch.randn(
            (args.batch, args.nvh, args.dk, args.dv), device=device, dtype=torch.float32
        ) / 8
    scale = args.dk ** -0.5

    swa_mask = torch.zeros((args.nvh), dtype=torch.bool, device=device)
    swa_mask[: int((args.swa_ratio * args.nvh + 0.999999))] = 1
    swa_mask = swa_mask[torch.randperm(args.nvh, device=device)]
    g[:, :, ~swa_mask] = 0.0
    print(
        f"device={torch.cuda.get_device_name()} sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    )
    print(f"shape=B{args.batch} T{args.tokens} Hg{args.nkh} H{args.nvh} D={args.dk}")
    print(f"swa_mask={swa_mask.to(torch.int32).tolist()}")

    g_ref, o_ref, a_ref, _h_ref, s_ref = ref_fwd(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        beta=beta.float(),
        scale=scale,
        initial_state=h0,
        cu_seqlens=None,
    )

    native_env = {
        "FLASHQLA_BLACKWELL_FWD_EXPERIMENT": None,
        "FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P": None,
        "FLASHQLA_BLACKWELL_SMALL_HV_PG_DTYPE": None,
        "FLASHQLA_BLACKWELL_SMALL_HV_USE_PG": None,
    }
    recompute_env = {
        "FLASHQLA_BLACKWELL_FWD_EXPERIMENT": "small_hv",
        "FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P": "1",
        "FLASHQLA_BLACKWELL_SMALL_HV_PG_DTYPE": None,
        "FLASHQLA_BLACKWELL_SMALL_HV_USE_PG": None,
    }
    precompute_p_env = {
        "FLASHQLA_BLACKWELL_FWD_EXPERIMENT": "small_hv",
        "FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P": None,
        "FLASHQLA_BLACKWELL_SMALL_HV_PG_DTYPE": None,
        "FLASHQLA_BLACKWELL_SMALL_HV_USE_PG": None,
    }
    pg_env = {
        "FLASHQLA_BLACKWELL_FWD_EXPERIMENT": "small_hv",
        "FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P": None,
        "FLASHQLA_BLACKWELL_SMALL_HV_PG_DTYPE": "fp32" if args.pg_fp32 else None,
        "FLASHQLA_BLACKWELL_SMALL_HV_USE_PG": "1",
    }

    _g_native, a_native, o_native, s_native = run_qla("native", q, k, v, g, beta, scale, h0, native_env)
    _g_rec, a_rec, o_rec, s_rec = run_qla("small_hv_recompute", q, k, v, g, beta, scale, h0, recompute_env)
    _g_pre, a_pre, o_pre, s_pre = run_qla("small_hv_precompute_p", q, k, v, g, beta, scale, h0, precompute_p_env)
    _g_pg, a_pg, o_pg, s_pg = run_qla("small_hv_pg", q, k, v, g, beta, scale, h0, pg_env)

    for repeat_idx in range(1, args.repeats):
        _g_pg_r, a_pg_r, o_pg_r, s_pg_r = run_qla(
            f"small_hv_pg.repeat{repeat_idx}", q, k, v, g, beta, scale, h0, pg_env
        )
        pg_o_diff = (o_pg_r.float() - o_rec.float()).abs()
        pg_s_diff = (s_pg_r.float() - s_rec.float()).abs()
        print(
            f"repeat{repeat_idx}: pg_vs_recompute_o_max={pg_o_diff.max().item():.6f} "
            f"pg_vs_recompute_s_max={pg_s_diff.max().item():.6f}"
        )
        if pg_o_diff.max().item() > o_ref.float().abs().amax().item() * 0.02:
            max_report(f"repeat{repeat_idx}.pg.o_vs_recompute", o_pg_r, o_rec, args.chunk_size)
            max_report(f"repeat{repeat_idx}.pg.s_vs_recompute", s_pg_r, s_rec, args.chunk_size)
            break

    print("\nAgainst reference:")
    max_report("native.o", o_native, o_ref, args.chunk_size)
    max_report("recompute.o", o_rec, o_ref, args.chunk_size)
    max_report("precompute_p.o", o_pre, o_ref, args.chunk_size)
    max_report("pg.o", o_pg, o_ref, args.chunk_size)
    max_report("native.s", s_native, s_ref, args.chunk_size)
    max_report("recompute.s", s_rec, s_ref, args.chunk_size)
    max_report("precompute_p.s", s_pre, s_ref, args.chunk_size)
    max_report("pg.s", s_pg, s_ref, args.chunk_size)

    print("\nVariant deltas:")
    max_report("precompute_p.o_vs_recompute", o_pre, o_rec, args.chunk_size)
    max_report("precompute_p.s_vs_recompute", s_pre, s_rec, args.chunk_size)
    max_report("pg.o_vs_recompute", o_pg, o_rec, args.chunk_size)
    max_report("pg.s_vs_recompute", s_pg, s_rec, args.chunk_size)
    max_report("recompute.o_vs_native", o_rec, o_native, args.chunk_size)
    max_report("precompute_p.o_vs_native", o_pre, o_native, args.chunk_size)
    max_report("pg.o_vs_native", o_pg, o_native, args.chunk_size)
    max_report("pg.A_vs_recompute", a_pg.float(), a_rec.float(), args.chunk_size)
    max_report("precompute_p.A_vs_recompute", a_pre.float(), a_rec.float(), args.chunk_size)
    max_report("native.A_vs_recompute", a_native.float(), a_rec.float(), args.chunk_size)


if __name__ == "__main__":
    main()
