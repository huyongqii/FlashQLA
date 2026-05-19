# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from typing import Optional

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_indices


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


def _alloc_inversion_buffers(block_S, accum_dtype, qkva_dtype):
    """Allocate the shared / fragment buffers used by the 64x64 lower-tri
    inversion. Returns a tuple in the order expected by
    ``_invert_64x64_lower_tri``: ``(a64_fragment, a16i_row, a16i_sum,
    a16i_shared, a16o_shared, a16o_fragment, a32i0_shared, a32i1_shared,
    a32o_shared, a32o_fragment, a64_shared)``.

    A tuple (rather than a dict) keeps the buffers as plain TileLang IR
    handles so they can be passed straight into a ``@T.macro`` -- a dict
    would force a Python ``__getitem__`` inside the macro body, which the
    AST parser cannot lower.
    """
    a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
    a16i_row = T.alloc_fragment((4, 16), dtype=accum_dtype)
    a16i_sum = T.alloc_fragment((4, 16), dtype=accum_dtype)
    a16i_shared = T.alloc_shared((4, 17, 16), dtype=accum_dtype)
    a16o_shared = T.alloc_shared((2, 17, 16), dtype=accum_dtype)
    a16o_fragment = T.alloc_fragment((2, 16, 16), dtype=accum_dtype)
    a32i0_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
    a32i1_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
    a32o_shared = T.alloc_shared((32, 32), dtype=accum_dtype)
    a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)
    a64_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
    T.annotate_layout(
        {
            a16i_shared: tilelang.layout.make_linear_layout(a16i_shared),
            a16o_shared: tilelang.layout.make_linear_layout(a16o_shared),
        }
    )
    return (
        a64_fragment,
        a16i_row,
        a16i_sum,
        a16i_shared,
        a16o_shared,
        a16o_fragment,
        a32i0_shared,
        a32i1_shared,
        a32o_shared,
        a32o_fragment,
        a64_shared,
    )


@T.macro
def _invert_64x64_lower_tri(
    block_S,
    a64_fragment,
    a16i_row,
    a16i_sum,
    a16i_shared,
    a16o_shared,
    a16o_fragment,
    a32i0_shared,
    a32i1_shared,
    a32o_shared,
    a32o_fragment,
    a64_shared,
):
    """Given ``a64_fragment`` holding ``I + StrictLower(beta * K @ K^T)``,
    compute its inverse and write it into ``a64_shared``.

    All scratch buffers must be allocated by the caller (so the layout
    annotations are visible at prim_func scope). This is a ``@T.macro`` so
    that TileLang's AST-level parser can recognise the
    ``for x, y in T.Parallel(...)`` constructs inside; using a plain Python
    function would let those expressions be evaluated eagerly and fail with
    ``'ForFrame' object is not iterable``.

    Algorithm:
      1. Split A into a 2x2 block matrix of 32x32 tiles. Each diagonal 32x32
         is itself split into 2x2 blocks of 16x16 tiles, so the four 16x16
         diagonal blocks (``a16i_shared``) are forward-substituted in place.
      2. First level: combine the two diagonal 16x16 inverses into the two
         32x32 inverses (``a32i0_shared`` / ``a32i1_shared``).
      3. Second level: combine the two 32x32 inverses with the off-diagonal
         block (``a32o_shared``) into the final 64x64 inverse stored in
         ``a64_shared``.
    """

    # Scatter A into the per-block staging buffers.
    #   - lower-left 32x32  -> a32o_shared (negated)
    #   - lower diagonal-of-32 16x16 (block (j_s//32, j_s//32)) off-diag pair
    #     -> a16o_shared (negated)
    #   - 4 diagonal 16x16 blocks -> a16i_shared
    for j_s, j_t in T.Parallel(block_S, block_S):
        if j_s >= 32 and j_t < 32:
            a32o_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t]
        elif (j_s // 16) == (j_t // 16) + 1:
            a16o_shared[j_s // 32, j_s % 16, j_t % 16] = -a64_fragment[j_s, j_t]
        elif (j_s // 16) == (j_t // 16):
            a16i_shared[j_s // 16, j_s % 16, j_t % 16] = a64_fragment[j_s, j_t]

    # Forward-substitute each 16x16 diagonal block in place.
    T.clear(a16i_row)
    for k_s in T.unroll(1, 16):
        for j_s, k_t in T.Parallel(4, 16):
            if k_t < k_s:
                a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
        T.clear(a16i_sum)
        for k_r in T.unroll(k_s):
            for j_s, k_t in T.Parallel(4, 16):
                a16i_sum[j_s, k_t] -= (
                    a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
                )
        for j_s, k_t in T.Parallel(4, 16):
            if k_t < k_s:
                a16i_shared[j_s, k_s, k_t] = a16i_sum[j_s, k_t]

    # First-level 2x16x16: combine two 16x16 diag inverses with the
    # off-diagonal 16x16 block to form a 32x32 inverse for each j_s.
    T.clear(a16o_fragment)
    for k_r in T.unroll(16):
        for j_s, k_s, k_t in T.Parallel(2, 16, 16):
            a16o_fragment[j_s, k_s, k_t] += (
                a16i_shared[j_s * 2 + 1, k_s, k_r] * a16o_shared[j_s, k_r, k_t]
            )
    for j_s, k_s, k_t in T.Parallel(2, 16, 16):
        a16o_shared[j_s, k_t, k_s] = a16o_fragment[j_s, k_s, k_t]
    T.clear(a16o_fragment)
    for k_r in T.unroll(16):
        for j_s, k_s, k_t in T.Parallel(2, 16, 16):
            a16o_fragment[j_s, k_s, k_t] += (
                a16o_shared[j_s, k_r, k_s] * a16i_shared[j_s * 2, k_r, k_t]
            )
    T.copy(a16o_fragment, a16o_shared[:, 0:16, 0:16])

    # Pack the two 32x32 inverses directly into a32i0_shared / a32i1_shared.
    # Skipping the previous fragment round-trip saves ~8KB of register file
    # per warpgroup; ``a32i?_shared`` are pure-shared and the GEMMs below
    # only consume them, so this is safe. Three mutually-exclusive passes
    # mirror the original layout (upper-right zero / lower-left from a16o /
    # diagonal from a16i) so each ``T.Parallel`` body remains a single
    # straight-line write.
    for k_s, k_t in T.Parallel(32, 32):
        if k_s < 16 and k_t >= 16:
            a32i0_shared[k_s, k_t] = 0
            a32i1_shared[k_s, k_t] = 0
    for k_s, k_t in T.Parallel(32, 32):
        if k_s >= 16 and k_t < 16:
            a32i0_shared[k_s, k_t] = a16o_shared[0, k_s - 16, k_t]
            a32i1_shared[k_s, k_t] = a16o_shared[1, k_s - 16, k_t]
    for k_s, k_t in T.Parallel(32, 32):
        if k_s // 16 == k_t // 16:
            a32i0_shared[k_s, k_t] = a16i_shared[k_s // 16, k_s % 16, k_t % 16]
            a32i1_shared[k_s, k_t] = a16i_shared[2 + k_s // 16, k_s % 16, k_t % 16]

    # Second-level 1x32x32. The small FP32 32x32 solves are not a supported
    # TCGEN05 operand pattern in TileLang 0.1.9, but hand-unrolled CUDA-core
    # loops are much slower on B200. Let TileLang choose its fastest lowering
    # for these two tiny GEMMs; the dominant K@K^T path stays explicit
    # TCGEN05/TMEM in the caller.
    T.gemm(a32i1_shared, a32o_shared, a32o_fragment, clear_accum=True)
    T.copy(a32o_fragment, a32o_shared)
    T.gemm(a32o_shared, a32i0_shared, a32o_fragment, clear_accum=True)

    # Combine the four 32x32 blocks into the final 64x64 inverse.
    #   * top-left  (rows 0..31,  cols 0..31)  <- a32i0_shared
    #   * bot-right (rows 32..63, cols 32..63) <- a32i1_shared
    #   * bot-left  (rows 32..63, cols 0..31)  <- a32o_fragment (last gemm)
    #   * top-right (rows 0..31,  cols 32..63) <- 0
    for k_s, k_t in T.Parallel(32, 32):
        a64_shared[k_s, k_t] = a32i0_shared[k_s, k_t]
    for k_s, k_t in T.Parallel(32, 32):
        a64_shared[32 + k_s, 32 + k_t] = a32i1_shared[k_s, k_t]
    for k_s, k_t in T.Parallel(32, 32):
        a64_shared[32 + k_s, k_t] = a32o_fragment[k_s, k_t]
    for k_s, k_t in T.Parallel(32, 32):
        a64_shared[k_s, 32 + k_t] = 0


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def tilelang_kkt_solve(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    seqlen_dtype,
    is_varlen,
):
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, num_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(
        bb,
        bh,
        bhg,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        k,
        b,
        a,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
        a64_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)

        (
            a64_fragment,
            a16i_row,
            a16i_sum,
            a16i_shared,
            a16o_shared,
            a16o_fragment,
            a32i0_shared,
            a32i1_shared,
            a32o_shared,
            a32o_fragment,
            a64_shared,
        ) = _alloc_inversion_buffers(block_S, accum_dtype, qkva_dtype)

        # ``data_is_ready`` is published by the K-load producer warp once
        # both ``k_shared`` and ``b_shared`` are populated, so the consumer
        # can fire ``tcgen05_gemm`` immediately on entry without first
        # going through a serial b-load.
        data_is_ready = T.alloc_barrier(arrive_count=32)
        a_is_ready = T.alloc_barrier(arrive_count=128)
        mma_mbar = T.alloc_barrier(arrive_count=1)

        tx = T.get_thread_binding()

        PRODUCER_NREG = 24
        CONSUMER_NREG = 64

        if tx < 128:
            T.set_max_nreg(CONSUMER_NREG, 1)

            T.barrier_wait(data_is_ready, 0)

            # A = K @ K^T (TCGEN05 / TMEM)
            T.tcgen05_gemm(
                k_shared,
                k_shared,
                a64_tmem,
                transpose_B=True,
                clear_accum=True,
                mbar=mma_mbar,
            )
            T.mbarrier_wait_parity(mma_mbar, 0)
            T.copy(a64_tmem, a64_fragment)

            # A <- I + StrictLower(b * A). Keep this as one pass over A:
            # the solve kernel is small enough that an extra 64x64 elementwise
            # sweep shows up in profiles.
            for j_s, j_t in T.Parallel(block_S, block_S):
                if j_s > j_t:
                    a64_fragment[j_s, j_t] *= b_shared[j_s]
                elif j_s == j_t:
                    a64_fragment[j_s, j_t] = 1
                else:
                    a64_fragment[j_s, j_t] = 0

            _invert_64x64_lower_tri(
                block_S,
                a64_fragment,
                a16i_row,
                a16i_sum,
                a16i_shared,
                a16o_shared,
                a16o_fragment,
                a32i0_shared,
                a32i1_shared,
                a32o_shared,
                a32o_fragment,
                a64_shared,
            )

            T.barrier_arrive(a_is_ready)

        else:
            T.set_max_nreg(PRODUCER_NREG, 0)

            if tx < 128 + 32:
                for j_s, j_k in T.Parallel(block_S, DK):
                    if left + j_s < seq_end_idx:
                        k_shared[j_s, j_k] = k[bb, left + j_s, bhg, j_k]
                    else:
                        k_shared[j_s, j_k] = 0
                for j_s in T.Parallel(block_S):
                    if left + j_s < seq_end_idx:
                        b_shared[j_s] = b[bb, left + j_s, bh]
                    else:
                        b_shared[j_s] = 0

                T.barrier_arrive(data_is_ready)

            else:
                # The remaining 96 producer threads jointly store A back
                # to gmem. Previously this was split into two warpgroups
                # (one for the unmasked path, one for the masked tail);
                # since exactly one branch executes per chunk the other
                # warpgroup was idle. Merging keeps the same semantics
                # while making the masked path run on 96 threads instead
                # of 64.
                T.barrier_wait(a_is_ready, 0)
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if left + j_s < seq_end_idx:
                        a[bb, left + j_s, bh, j_t] = a64_shared[j_s, j_t]

    if is_varlen:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
        ):
            with T.Kernel(num_chunks * H, threads=256) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    bh,
                    bhg,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    else:

        @T.prim_func
        def tilelang_kkt_solve_kernel(
            k: T.Tensor(k_shape, dtype=qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            num_chunks: T.int32,
        ):
            with T.Kernel(num_chunks * H, threads=256) as (bch,):
                bc, bh = bch // H, bch % H
                bhg = bh // (H // Hg)

                bb = bc % data_batch_size
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    bh,
                    bhg,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    k,
                    b,
                    a,
                )

    return tilelang_kkt_solve_kernel


def kkt_solve(
    k: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H = b.shape
    assert K == 128
    assert chunk_size == 64

    if cu_seqlens is None:
        num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        if batch_size != 1:
            raise ValueError(
                "Blackwell KKT varlen expects packed inputs with batch dimension 1 "
                f"when cu_seqlens is provided, got batch_size={batch_size}."
            )
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype = cu_seqlens.dtype
        is_varlen = True

    a = torch.empty(
        (batch_size, num_tokens, H, chunk_size), dtype=k.dtype, device=k.device
    )
    kernel = tilelang_kkt_solve(
        H,
        Hg,
        K,
        chunk_size,
        qkva_dtype=k.dtype,
        b_dtype=b.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        is_varlen=is_varlen,
    )
    if is_varlen:
        kernel(k, b, cu_seqlens, chunk_indices, a)
    else:
        kernel(k, b, a, num_chunks)

    return a
