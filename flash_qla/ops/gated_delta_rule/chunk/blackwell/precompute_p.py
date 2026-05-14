# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_precompute_p(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qk_dtype,
    p_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    p_shape = (batch_size, num_tokens, H, chunk_size)

    @T.prim_func
    def tilelang_precompute_p_kernel(
        q: T.Tensor(q_shape, dtype=qk_dtype),
        k: T.Tensor(k_shape, dtype=qk_dtype),
        p: T.Tensor(p_shape, dtype=p_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(num_chunks * H, threads=256) as (bch,):
            bc, bh = bch // H, bch % H
            bhg = bh // (H // Hg)
            bb = bc % batch_size
            chunk_idx = bc // batch_size
            left = chunk_idx * block_S
            right = left + block_S

            q_shared = T.alloc_shared((block_S, DK), dtype=qk_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qk_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            T.copy(q[bb, left:right, bhg, 0:DK], q_shared)
            T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
            T.gemm(q_shared, k_shared, p_fragment, transpose_B=True, clear_accum=True)
            T.copy(p_fragment, p[bb, left:right, bh, 0:block_S])

    return tilelang_precompute_p_kernel


def precompute_p(
    q: torch.Tensor,
    k: torch.Tensor,
    num_v_heads: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    batch_size, num_tokens, Hg, K = k.shape
    assert K == 128
    assert chunk_size == 64
    num_chunks = batch_size * tilelang.cdiv(num_tokens, chunk_size)
    p = torch.empty(
        (batch_size, num_tokens, num_v_heads, chunk_size),
        dtype=q.dtype,
        device=q.device,
    )
    kernel = tilelang_precompute_p(
        num_v_heads,
        Hg,
        K,
        chunk_size,
        accum_dtype="float32",
        qk_dtype=q.dtype,
        p_dtype=p.dtype,
    )
    kernel(q, k, p, num_chunks)
    return p
