#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Check TileLang Blackwell precomputed Pg materialization."""

from __future__ import annotations

import argparse

import torch

from flash_qla.ops.gated_delta_rule.chunk.blackwell.fused_fwd_native import (
    tilelang_precompute_pg_blackwell,
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
    g = g_raw.reshape(args.batch, -1, args.chunk_size, args.nvh).cumsum(dim=2)
    g = g.reshape(args.batch, args.tokens, args.nvh)
    pg = torch.empty(
        args.batch,
        args.tokens,
        args.nvh,
        args.chunk_size,
        device=device,
        dtype=dtype,
    )
    num_chunks = args.tokens // args.chunk_size

    kernel = tilelang_precompute_pg_blackwell(
        args.nvh,
        args.nkh,
        args.dk,
        args.chunk_size,
        scale,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
    )
    kernel(q, k, g, pg, num_chunks)
    torch.cuda.synchronize()

    q_chunks = q.reshape(args.batch, num_chunks, args.chunk_size, args.nkh, args.dk)
    k_chunks = k.reshape_as(q_chunks)
    heads_per_group = args.nvh // args.nkh
    ref = torch.empty_like(pg, dtype=torch.float32)
    for b in range(args.batch):
        for c in range(num_chunks):
            for hv in range(args.nvh):
                hg = hv // heads_per_group
                p = q_chunks[b, c, :, hg, :].float() @ k_chunks[b, c, :, hg, :].float().T
                g_chunk = g[
                    b,
                    c * args.chunk_size : (c + 1) * args.chunk_size,
                    hv,
                ]
                g_mat = torch.exp(g_chunk[:, None] - g_chunk[None, :])
                g_mat = torch.tril(g_mat)
                ref[
                    b,
                    c * args.chunk_size : (c + 1) * args.chunk_size,
                    hv,
                    :,
                ] = p * g_mat * scale

    abs_err, rel_err = _max_err(pg.float(), ref)
    trans_ref = ref.reshape(
        args.batch, num_chunks, args.chunk_size, args.nvh, args.chunk_size
    ).transpose(2, 4).reshape_as(ref)
    trans_abs, trans_rel = _max_err(pg.float(), trans_ref)

    print(
        f"device={torch.cuda.get_device_name()} "
        f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    )
    print(
        f"shape=B{args.batch} T{args.tokens} Hg{args.nkh} H{args.nvh} "
        f"DK{args.dk} chunk{args.chunk_size} dtype={args.dtype}"
    )
    print(f"pg_abs={abs_err:.6f} pg_rel={rel_err:.6f}")
    print(f"pg_transposed_ref_abs={trans_abs:.6f} pg_transposed_ref_rel={trans_rel:.6f}")
    print(f"pg_amax={pg.abs().amax().item():.6f} ref_amax={ref.abs().amax().item():.6f}")
    if abs_err < trans_abs:
        print("RESULT: precompute_pg layout looks row-major")
    else:
        print("RESULT: precompute_pg output looks closer to transposed layout")


if __name__ == "__main__":
    main()
