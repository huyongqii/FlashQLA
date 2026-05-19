#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Smoke test for the 2-WG + multi-TMEM redesign of fused_fwd_native.

Validates three preconditions that the redesign depends on:

1. TMEM accumulator persistence across GEMMs
   ``c_tmem = A0 @ B0`` (clear_accum=True), then
   ``c_tmem += A1 @ B1`` (clear_accum=False)
   Result must equal ``A0@B0 + A1@B1``.

2. Multiple coexisting ``T.alloc_tmem`` in the same kernel, both written
   by ``tcgen05_gemm`` (this is the only legal write path: TileLang's
   TMEM layout is established by the gemm tile op, so non-gemm writes
   like T.copy(frag -> tmem) without a prior gemm fail to compile).

3. Producer / MM WG split with mbarrier handshake
   Thread block is partitioned into a producer WG (loads A/B into shared)
   and an MM WG that waits on a barrier and then issues tcgen05 GEMMs.

If any of the three checks fails, the 6-hour redesign should NOT proceed
without first working around the missing TileLang capability.
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def smoke_kernel(M: int, N: int, K: int):
    """2-WG kernel exercising multi-TMEM + cross-WG handshake + accumulation.

    Layout:
      tx <  128 : MM WG. Issues two tcgen05_gemm into c_tmem (the second
                  with ``clear_accum=False`` to test accumulation), and one
                  independent tcgen05_gemm into d_tmem (multi-TMEM check).
      tx >= 128 : Producer WG. Single warp loads A0/B0, then A1/B1, signaling
                  the MM WG via mbarriers (cross-WG handshake check).
    """
    @T.prim_func
    def kernel(
        a0: T.Tensor((M, K), dtype="bfloat16"),
        b0: T.Tensor((K, N), dtype="bfloat16"),
        a1: T.Tensor((M, K), dtype="bfloat16"),
        b1: T.Tensor((K, N), dtype="bfloat16"),
        # c = a0@b0 + a1@b1   (accumulation test)
        c: T.Tensor((M, N), dtype="float32"),
        # d = a1@b1            (independent second TMEM)
        d: T.Tensor((M, N), dtype="float32"),
    ):
        with T.Kernel(1, threads=160) as (_,):
            a0_shared = T.alloc_shared((M, K), dtype="bfloat16")
            b0_shared = T.alloc_shared((K, N), dtype="bfloat16")
            a1_shared = T.alloc_shared((M, K), dtype="bfloat16")
            b1_shared = T.alloc_shared((K, N), dtype="bfloat16")

            # Two coexisting TMEM regions (each populated by tcgen05_gemm).
            c_tmem = T.alloc_tmem((M, N), dtype="float32")
            d_tmem = T.alloc_tmem((M, N), dtype="float32")

            c_frag = T.alloc_fragment((M, N), dtype="float32")
            d_frag = T.alloc_fragment((M, N), dtype="float32")

            # Cross-WG handshake: producer signals when each tile is ready.
            ab0_ready = T.alloc_barrier(arrive_count=32)
            ab1_ready = T.alloc_barrier(arrive_count=32)
            # tcgen05 completion barriers (issued by MM WG, awaited by MM WG).
            mma_c0_done = T.alloc_barrier(arrive_count=1)
            mma_c1_done = T.alloc_barrier(arrive_count=1)
            mma_d_done = T.alloc_barrier(arrive_count=1)

            tx = T.get_thread_binding()

            if tx < 128:
                # --- MM WG ---
                # Wait for tile 0, then issue C = A0 @ B0 (clear).
                T.barrier_wait(ab0_ready, 0)
                T.tcgen05_gemm(
                    a0_shared, b0_shared, c_tmem,
                    clear_accum=True,
                    mbar=mma_c0_done,
                )
                # Independent D = A1 @ B1 launched in parallel; uses a1_shared
                # which the producer fills next, so we wait ab1_ready first.
                T.barrier_wait(ab1_ready, 0)
                T.tcgen05_gemm(
                    a1_shared, b1_shared, d_tmem,
                    clear_accum=True,
                    mbar=mma_d_done,
                )
                # Accumulation test: C += A1 @ B1 (clear_accum=False).
                # Must wait c0 first so the accumulator is settled.
                T.mbarrier_wait_parity(mma_c0_done, 0)
                T.tcgen05_gemm(
                    a1_shared, b1_shared, c_tmem,
                    clear_accum=False,
                    mbar=mma_c1_done,
                )

                T.mbarrier_wait_parity(mma_c1_done, 0)
                T.mbarrier_wait_parity(mma_d_done, 0)
                T.copy(c_tmem, c_frag)
                T.copy(d_tmem, d_frag)
                T.copy(c_frag, c)
                T.copy(d_frag, d)
            else:
                # --- Producer WG (single warp = 32 threads) ---
                if tx < 160:
                    T.copy(a0, a0_shared)
                    T.copy(b0, b0_shared)
                    T.barrier_arrive(ab0_ready)

                    T.copy(a1, a1_shared)
                    T.copy(b1, b1_shared)
                    T.barrier_arrive(ab1_ready)

    return kernel


def _check(name: str, got: torch.Tensor, ref: torch.Tensor, rtol: float = 0.05) -> bool:
    err = (got - ref).abs().max().item()
    denom = ref.abs().max().item() + 1e-6
    rel = err / denom
    ok = rel < rtol
    flag = "OK" if ok else "FAIL"
    print(f"[{flag}] {name:20s} max_abs={err:.6f}  rel={rel:.6f}  ref_max={denom:.4f}")
    return ok


def main() -> int:
    torch.manual_seed(0)
    M, N, K = 64, 64, 128

    a0 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b0 = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)
    a1 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b1 = torch.randn((K, N), device="cuda", dtype=torch.bfloat16)

    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    d = torch.empty((M, N), device="cuda", dtype=torch.float32)

    ref_c = a0.float() @ b0.float() + a1.float() @ b1.float()
    ref_d = a1.float() @ b1.float()

    print("Compiling smoke kernel...")
    kernel = smoke_kernel(M, N, K)
    print("Launching...")
    kernel(a0, b0, a1, b1, c, d)
    torch.cuda.synchronize()
    print("Done. Checking results:")

    ok_c = _check("tmem accum (c)", c, ref_c)
    ok_d = _check("multi-tmem (d)", d, ref_d)

    print()
    if ok_c and ok_d:
        print("RESULT: ALL PRECONDITIONS PASS - safe to start the 2-WG + TMEM redesign.")
        return 0
    else:
        print("RESULT: at least one precondition FAILED. Do NOT start the redesign yet.")
        if not ok_c:
            print("  - TMEM accumulation (clear_accum=False) is broken or not supported.")
        if not ok_d:
            print("  - A second coexisting alloc_tmem does not work; redesign needs rework.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
