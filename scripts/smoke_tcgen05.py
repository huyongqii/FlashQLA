#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Minimal TileLang TCGEN05 smoke test for B200/B300."""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


@tilelang.jit()
def tcgen05_smoke_kernel(M: int, N: int, K: int):
    @T.prim_func
    def kernel(
        a: T.Tensor((M, K), dtype="bfloat16"),
        b: T.Tensor((K, N), dtype="bfloat16"),
        c: T.Tensor((M, N), dtype="float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((M, K), dtype="bfloat16")
            b_shared = T.alloc_shared((K, N), dtype="bfloat16")
            c_tmem = T.alloc_tmem((M, N), dtype="float32")
            c_frag = T.alloc_fragment((M, N), dtype="float32")
            mbar = T.alloc_barrier(arrive_count=1)

            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.tcgen05_gemm(
                a_shared,
                b_shared,
                c_tmem,
                clear_accum=True,
                mbar=mbar,
            )
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(c_tmem, c_frag)
            T.copy(c_frag, c)

    return kernel


def main() -> int:
    torch.manual_seed(0)
    M, N, K = 64, 64, 128
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    ref = a.float() @ b.float()

    kernel = tcgen05_smoke_kernel(M, N, K)
    kernel(a, b, c)
    torch.cuda.synchronize()

    err = (c - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"max_err={err:.6f} rel={rel:.6f}")
    return 0 if rel < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
