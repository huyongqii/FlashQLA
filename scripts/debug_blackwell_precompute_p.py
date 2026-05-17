#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Check Blackwell precomputed P = Q @ K^T materialization.

This isolates the small-Hv P-reuse path from the full GDR recurrence.  It
compares the TileLang precompute kernel output against a torch reference and
also reports common layout mistakes such as transposed chunk matrices.
"""

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
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if args.tokens % args.chunk_size != 0:
        raise ValueError("--tokens must be divisible by --chunk-size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    q = torch.randn(
        args.batch, args.tokens, args.nkh, args.dk, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
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
    ref = torch.empty_like(p)
    for b in range(args.batch):
        for c in range(num_chunks):
            for h in range(args.nkh):
                ref[
                    b,
                    c * args.chunk_size : (c + 1) * args.chunk_size,
                    h,
                    :,
                ] = q_chunks[b, c, :, h, :].float() @ k_chunks[b, c, :, h, :].float().T

    abs_err, rel_err = _max_err(p, ref)
    transposed_ref = ref.reshape(
        args.batch, num_chunks, args.chunk_size, args.nkh, args.chunk_size
    ).transpose(2, 4).reshape_as(ref)
    trans_abs, trans_rel = _max_err(p, transposed_ref)

    print(
        f"device={torch.cuda.get_device_name()} "
        f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
    )
    print(
        f"shape=B{args.batch} T{args.tokens} Hg{args.nkh} "
        f"DK{args.dk} chunk{args.chunk_size} dtype={args.dtype}"
    )
    print(f"direct_ref_abs={abs_err:.6f} direct_ref_rel={rel_err:.6f}")
    print(f"transposed_ref_abs={trans_abs:.6f} transposed_ref_rel={trans_rel:.6f}")
    print(f"p_amax={p.abs().amax().item():.6f} ref_amax={ref.abs().amax().item():.6f}")

    if abs_err < trans_abs:
        print("RESULT: precompute_p layout looks row-major")
    else:
        print("RESULT: precompute_p output looks closer to transposed layout")


if __name__ == "__main__":
    main()
