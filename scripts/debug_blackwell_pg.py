#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Check Blackwell Pg = scale * G * (Q @ K^T) materialization."""

from __future__ import annotations

import argparse

import torch

from flash_qla.ops.gated_delta_rule.chunk.blackwell.fused_fwd_native import (
    tilelang_precompute_p_blackwell,
)


def _max_err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().amax().clamp_min(1e-6)
    return diff.amax().item(), (diff.amax() / denom).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--nkh", type=int, default=2)
    parser.add_argument("--nvh", type=int, default=8)
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if args.tokens % args.chunk_size != 0:
        raise ValueError("--tokens must be divisible by --chunk-size")
    if args.nvh % args.nkh != 0:
        raise ValueError("--nvh must be divisible by --nkh")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    scale = args.dk ** -0.5

    q = torch.randn(
        args.batch, args.tokens, args.nkh, args.dk, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    g_raw = torch.randn(
        args.batch, args.tokens, args.nvh, device=device, dtype=torch.float32
    )
    # Mirror chunk_local_cumsum only enough for Pg layout/value checking.
    g = g_raw.reshape(args.batch, -1, args.chunk_size, args.nvh).cumsum(dim=2)
    g = g.reshape(args.batch, args.tokens, args.nvh)

    p = torch.empty(
        args.batch,
        args.tokens,
        args.nkh,
        args.chunk_size,
        device=device,
        dtype=torch.float32,
    )
    num_chunks = args.tokens // args.chunk_size

    kernel = tilelang_precompute_p_blackwell(
        args.nkh,
        args.dk,
        args.chunk_size,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
    )
    kernel(q, k, p, num_chunks)
    torch.cuda.synchronize()

    q_chunks = q.reshape(args.batch, num_chunks, args.chunk_size, args.nkh, args.dk)
    k_chunks = k.reshape_as(q_chunks)
    p_ref = torch.empty_like(p)
    for b in range(args.batch):
        for c in range(num_chunks):
            for h in range(args.nkh):
                p_ref[
                    b,
                    c * args.chunk_size : (c + 1) * args.chunk_size,
                    h,
                    :,
                ] = q_chunks[b, c, :, h, :].float() @ k_chunks[b, c, :, h, :].float().T

    pg_abs_max = 0.0
    pg_rel_max = 0.0
    pg_trans_abs_max = 0.0
    pg_trans_rel_max = 0.0
    heads_per_group = args.nvh // args.nkh
    for hv in range(args.nvh):
        hg = hv // heads_per_group
        g_h = g[:, :, hv].reshape(args.batch, num_chunks, args.chunk_size)
        g_diff = g_h[:, :, :, None] - g_h[:, :, None, :]
        g_mask = torch.tril(torch.ones(args.chunk_size, args.chunk_size, device=device))
        g_mat = torch.exp(g_diff).masked_fill(g_mask[None, None, :, :] == 0, 0)
        p_h = p[:, :, hg, :].reshape(args.batch, num_chunks, args.chunk_size, args.chunk_size)
        p_ref_h = p_ref[:, :, hg, :].reshape_as(p_h)
        pg = p_h * g_mat * scale
        pg_ref = p_ref_h * g_mat * scale
        abs_err, rel_err = _max_err(pg, pg_ref)
        pg_abs_max = max(pg_abs_max, abs_err)
        pg_rel_max = max(pg_rel_max, rel_err)

        pg_trans_ref = p_ref_h.transpose(-1, -2) * g_mat * scale
        trans_abs, trans_rel = _max_err(pg, pg_trans_ref)
        pg_trans_abs_max = max(pg_trans_abs_max, trans_abs)
        pg_trans_rel_max = max(pg_trans_rel_max, trans_rel)

    p_abs, p_rel = _max_err(p, p_ref)
    print(
        f"device={torch.cuda.get_device_name()} "
        f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    )
    print(
        f"shape=B{args.batch} T{args.tokens} Hg{args.nkh} H{args.nvh} "
        f"DK{args.dk} chunk{args.chunk_size} dtype={args.dtype}"
    )
    print(f"p_abs={p_abs:.6f} p_rel={p_rel:.6f}")
    print(f"pg_abs={pg_abs_max:.6f} pg_rel={pg_rel_max:.6f}")
    print(f"pg_transposed_ref_abs={pg_trans_abs_max:.6f} pg_transposed_ref_rel={pg_trans_rel_max:.6f}")
    if pg_abs_max < pg_trans_abs_max:
        print("RESULT: Pg from precomputed P matches recomputed row-major reference")
    else:
        print("RESULT: Pg is closer to transposed reference")


if __name__ == "__main__":
    main()
