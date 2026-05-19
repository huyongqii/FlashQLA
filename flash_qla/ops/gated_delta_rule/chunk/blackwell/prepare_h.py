# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import torch
import tilelang
import tilelang.language as T

from flash_qla.utils import prepare_chunk_offsets


# Blackwell CP state summary kernel.


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_prepare_h(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    h_dtype,
    mask_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    store_h,
    store_final_correction,
    use_fallback_mask,
    is_varlen,
    is_cp,
    num_stages=2,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    if is_varlen:
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        h_shape = (1, num_chunks, H, DK, DV)
    else:
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        h_shape = (batch_size, num_chunks, H, DK, DV)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    m_shape = (batch_size, H, DK, DK)
    fallback_mask_shape = (batch_size, H)

    @T.prim_func
    def tilelang_prepare_h_kernel(
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        num_warmup_chunks: T.Tensor([batch_size, H], dtype=seqlen_dtype),
        fallback_mask: T.Tensor(fallback_mask_shape, dtype=mask_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
        mt: T.Tensor(m_shape, dtype=ht_dtype),
    ):
        with T.Kernel(batch_size * H, threads=512) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            _seq_split_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")
            _chunk_split_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            num_iters = T.alloc_var("int32")
            num_iters = (
                num_warmup_chunks[bb, bh]
                if is_cp
                else T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            )

            calc_mt = T.alloc_var("bool")
            if store_final_correction:
                calc_mt = is_cp and num_iters >= T.ceildiv(
                    seq_end_idx - seq_start_idx, block_S
                )
                if use_fallback_mask:
                    calc_mt = calc_mt and fallback_mask[bb, bh]
            else:
                calc_mt = False
            seq_start_idx = (
                seq_end_idx - num_iters * block_S if is_cp else seq_start_idx
            )

            k_shared = T.alloc_shared((num_stages, block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((num_stages, block_S, DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((num_stages, block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared(
                (num_stages, block_S), dtype=accum_dtype, scope="shared"
            )
            b_shared = T.alloc_shared(
                (num_stages, block_S), dtype=accum_dtype, scope="shared"
            )
            h_shared = T.alloc_shared((DK, DV), dtype=qkva_dtype)
            x_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            y_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            if store_final_correction:
                m_shared_L = T.alloc_shared((DK, DK // 2), dtype=qkva_dtype)
                m_shared_R = T.alloc_shared((DK, DK // 2), dtype=qkva_dtype)
                z_shared_L = T.alloc_shared((block_S, DK // 2), dtype=qkva_dtype)
                z_shared_R = T.alloc_shared((block_S, DK // 2), dtype=qkva_dtype)
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            x_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            y_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            if store_final_correction:
                m_fragment_L = T.alloc_fragment((DK, DK // 2), dtype=accum_dtype)
                m_fragment_R = T.alloc_fragment((DK, DK // 2), dtype=accum_dtype)
                z_fragment_L = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
                z_fragment_R = T.alloc_fragment((block_S, DK // 2), dtype=accum_dtype)
            g_last_local_S = T.alloc_local((1), dtype=accum_dtype)
            g_last_local_Y = T.alloc_local((1), dtype=accum_dtype)
            if store_final_correction:
                g_last_local_X = T.alloc_local((1), dtype=accum_dtype)
                g_prod_X = T.alloc_fragment((1), dtype=accum_dtype)
                g_prod_Y = T.alloc_fragment((1), dtype=accum_dtype)

            data_is_ready = T.alloc_barrier(
                arrive_count=[(96 if store_h else 128)] * num_stages
            )
            data_is_free = T.alloc_barrier(arrive_count=[384] * num_stages)

            bar_0 = T.alloc_barrier(arrive_count=(416 if store_h else 384))
            bar_1 = T.alloc_barrier(arrive_count=256)
            bar_2 = T.alloc_barrier(arrive_count=384)
            if store_final_correction:
                bar_3 = T.alloc_barrier(arrive_count=128)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_NREG = 24
            CONSUMER_S_NREG = 168
            CONSUMER_X_NREG = 160
            CONSUMER_Y_NREG = 160

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                # Initialize S
                if use_initial_state:
                    T.copy(h0[bb, bh, 0:DK, 0:DV], h_fragment)
                else:
                    T.clear(h_fragment)

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE = i_s % num_stages]
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2
                    )
                    T.barrier_arrive(bar_0)

                    # [STAGE = i_s % num_stages] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # S4[1] S
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    # [STAGE = i_s % num_stages] 1
                    T.barrier_wait(bar_1, i_s % 2)
                    # S = g_last * S
                    g_last_local_S[0] = T.exp2(
                        g_shared[i_s % num_stages, block_S - 1] * 1.442695
                    )
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] *= g_last_local_S[0]
                    T.barrier_arrive(bar_2)

                    # [STAGE = i_s % num_stages] 2
                    T.barrier_wait(bar_2, i_s % 2)
                    # S += X^T @ Y
                    T.gemm(
                        x_shared,
                        y_shared,
                        h_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    if store_final_correction:
                        T.barrier_arrive(bar_3)

                    T.barrier_arrive(data_is_free[i_s % num_stages])

                # Store final S
                if store_final_state:
                    T.copy(h_fragment, ht[bb, bh, 0:DK, 0:DV])

            elif tx < 256:
                T.set_max_nreg(CONSUMER_X_NREG, 1)

                if store_final_correction:
                    if calc_mt:
                        for j_k, j_v in T.Parallel(DK, DK // 2):
                            if j_k == j_v + DK // 2:
                                m_fragment_R[j_k, j_v] = 1
                            else:
                                m_fragment_R[j_k, j_v] = 0
                        g_prod_X[0] = 0

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE = i_s % num_stages]
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2
                    )
                    T.barrier_arrive(bar_0)

                    # [STAGE = i_s % num_stages] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # X = A^T @ K
                    T.gemm(
                        a_shared[i_s % num_stages, :, :],
                        k_shared[i_s % num_stages, :, :],
                        x_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )

                    # [STAGE = i_s % num_stages] 1
                    # X = - b * X
                    for j_s, j_k in T.Parallel(block_S, DK):
                        x_shared[j_s, j_k] = (
                            x_fragment[j_s, j_k]
                            * -b_shared[i_s % num_stages, j_s]
                        )
                    T.barrier_arrive(bar_2)

                    if store_final_correction:
                        if calc_mt:
                            # [STAGE = i_s % num_stages] 2
                            g_prod_X[0] += g_shared[i_s % num_stages, block_S - 1]
                            # S4[2] M
                            T.copy(m_fragment_R, m_shared_R)

                            # [STAGE = i_s % num_stages] 3
                            T.barrier_wait(bar_3, i_s % 2)
                            # Z = K @ M
                            T.gemm(
                                k_shared[i_s % num_stages, :, :],
                                m_shared_R,
                                z_fragment_R,
                                clear_accum=True,
                            )
                            # S4[2] Z
                            T.copy(z_fragment_R, z_shared_R)
                            # M += X^T @ Z
                            T.gemm(
                                x_shared,
                                z_shared_R,
                                m_fragment_R,
                                transpose_A=True,
                                clear_accum=False,
                            )

                    T.barrier_arrive(data_is_free[i_s % num_stages])

                if store_final_correction:
                    if calc_mt:
                        g_last_local_X[0] = T.exp2(g_prod_X[0] * 1.442695)
                        for j_k, j_v in T.Parallel(DK, DK // 2):
                            m_fragment_R[j_k, j_v] *= g_last_local_X[0]
                        T.copy(m_fragment_R, mt[bb, bh, 0:DK, DK // 2 :])

            elif tx < 384:
                T.set_max_nreg(CONSUMER_Y_NREG, 1)

                if store_final_correction:
                    if calc_mt:
                        for j_k, j_v in T.Parallel(DK, DK // 2):
                            if j_k == j_v:
                                m_fragment_L[j_k, j_v] = 1
                            else:
                                m_fragment_L[j_k, j_v] = 0
                        g_prod_Y[0] = 0

                # Main Loop
                for i_s in T.serial(num_iters):
                    # [STAGE = i_s % num_stages]
                    T.barrier_wait(
                        data_is_ready[i_s % num_stages], (i_s // num_stages + 0) % 2
                    )
                    T.barrier_arrive(bar_0)

                    # [STAGE = i_s % num_stages] 0
                    T.barrier_wait(bar_0, i_s % 2)
                    # Precompute g_last/g
                    g_last_local_Y[0] = g_shared[i_s % num_stages, block_S - 1]
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_last_local_Y[0] - g_shared[i_s % num_stages, j_s])
                            * 1.442695
                        )
                    g_last_local_Y[0] = T.exp2(g_last_local_Y[0] * 1.442695)
                    T.barrier_arrive(bar_1)

                    # [STAGE = i_s % num_stages] 1
                    T.barrier_wait(bar_1, i_s % 2)
                    # U = K @ S
                    T.gemm(
                        k_shared[i_s % num_stages, :, :],
                        h_shared,
                        y_fragment,
                        clear_accum=True,
                    )
                    # Y = g_last * U - g_last/g * V
                    for j_s, j_v in T.Parallel(block_S, DV):
                        y_shared[j_s, j_v] = (
                            y_fragment[j_s, j_v] * g_last_local_Y[0]
                            - v_shared[i_s % num_stages, j_s, j_v]
                            * g_rev_exp_shared[j_s]
                        )
                    T.barrier_arrive(bar_2)

                    if store_final_correction:
                        if calc_mt:
                            # [STAGE = i_s % num_stages] 2
                            g_prod_Y[0] += g_shared[i_s % num_stages, block_S - 1]
                            # S4[2] M
                            T.copy(m_fragment_L, m_shared_L)

                            # [STAGE = i_s % num_stages] 3
                            T.barrier_wait(bar_3, i_s % 2)
                            # Z = K @ M
                            T.gemm(
                                k_shared[i_s % num_stages, :, :],
                                m_shared_L,
                                z_fragment_L,
                                clear_accum=True,
                            )
                            # S4[2] Z
                            T.copy(z_fragment_L, z_shared_L)
                            # M += X^T @ Z
                            T.gemm(
                                x_shared,
                                z_shared_L,
                                m_fragment_L,
                                transpose_A=True,
                                clear_accum=False,
                            )

                    T.barrier_arrive(data_is_free[i_s % num_stages])

                if store_final_correction:
                    if calc_mt:
                        g_last_local_Y[0] = T.exp2(g_prod_Y[0] * 1.442695)
                        for j_k, j_v in T.Parallel(DK, DK // 2):
                            m_fragment_L[j_k, j_v] *= g_last_local_Y[0]
                        T.copy(m_fragment_L, mt[bb, bh, 0:DK, : DK // 2])

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2
                        )
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load K
                        T.copy(
                            k[batch_idx, left:right, bhg, 0:DK],
                            k_shared[i_s % num_stages, :, :],
                        )

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2
                        )
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load V
                        T.copy(
                            v[batch_idx, left:right, bh, 0:DV],
                            v_shared[i_s % num_stages, :, :],
                        )
                        if store_h:
                            # Keep the original 3-warp producer schedule when
                            # the final store warp is occupied by output_h.
                            T.copy(
                                a[batch_idx, left:right, bh, 0:block_S],
                                a_shared[i_s % num_stages, :, :],
                            )

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(
                            data_is_free[i_s % num_stages], (i_s // num_stages + 1) % 2
                        )
                        left = seq_start_idx + i_s * block_S
                        right = left + block_S

                        # Load gamma
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared[i_s % num_stages, j_s] = g[
                                    batch_idx, left + j_s, bh
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    g_shared[i_s % num_stages, j_s] = g[
                                        batch_idx, seq_end_idx - 1, bh
                                    ]
                        # Load beta
                        if right <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared[i_s % num_stages, j_s] = b[
                                    batch_idx, left + j_s, bh
                                ]
                        else:
                            for j_s in T.Parallel(block_S):
                                if left + j_s < seq_end_idx:
                                    b_shared[i_s % num_stages, j_s] = b[
                                        batch_idx, left + j_s, bh
                                    ]
                                else:
                                    b_shared[i_s % num_stages, j_s] = 0

                        T.barrier_arrive(data_is_ready[i_s % num_stages])

                else:
                    if store_h:
                        for i_s in T.serial(num_iters):
                            T.barrier_arrive(bar_0)

                            T.barrier_wait(bar_0, i_s % 2)
                            T.barrier_wait(bar_1, i_s % 2)
                            # Store S
                            T.copy(
                                h_shared,
                                h[batch_idx, chunk_start_idx + i_s, bh, 0:DK, 0:DV],
                            )
                    else:
                        for i_s in T.serial(num_iters):
                            T.barrier_wait(
                                data_is_free[i_s % num_stages],
                                (i_s // num_stages + 1) % 2,
                            )
                            left = seq_start_idx + i_s * block_S
                            right = left + block_S

                            T.copy(
                                a[batch_idx, left:right, bh, 0:block_S],
                                a_shared[i_s % num_stages, :, :],
                            )

                            T.barrier_arrive(data_is_ready[i_s % num_stages])

    return tilelang_prepare_h_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_prepare_h_cp_v2(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    block_DV,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    mask_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_correction,
    use_fallback_mask,
    is_varlen,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size
    num_DV_blocks = T.ceildiv(DV, block_DV)

    if is_varlen:
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
    else:
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    m_shape = (batch_size, H, DK, DK)
    fallback_mask_shape = (batch_size, H)

    @T.prim_func
    def tilelang_prepare_h_cp_v2_kernel(
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        num_warmup_chunks: T.Tensor([batch_size, H], dtype=seqlen_dtype),
        fallback_mask: T.Tensor(fallback_mask_shape, dtype=mask_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
        mt: T.Tensor(m_shape, dtype=ht_dtype),
    ):
        with T.Kernel(batch_size * H * num_DV_blocks, threads=384) as (bbhv,):
            bbh, bv = bbhv // num_DV_blocks, bbhv % num_DV_blocks
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            num_iters = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            num_iters = num_warmup_chunks[bb, bh]
            seq_start_idx = seq_end_idx - num_iters * block_S

            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            x_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            y_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            if store_final_correction:
                m_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
                z_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            x_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            y_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            if store_final_correction:
                m_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
                z_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            g_last_local_S = T.alloc_local((1), dtype=accum_dtype)
            g_last_local_Y = T.alloc_local((1), dtype=accum_dtype)
            if store_final_correction:
                g_prod_M = T.alloc_local((1), dtype=accum_dtype)

            data_loaded = T.alloc_barrier(arrive_count=384)
            bar_1 = T.alloc_barrier(arrive_count=256)
            bar_2 = T.alloc_barrier(arrive_count=384)
            if store_final_correction:
                state_ready = T.alloc_barrier(arrive_count=128)
            iter_done = T.alloc_barrier(arrive_count=384)

            T.use_swizzle(10)

            tx = T.get_thread_binding()

            CONSUMER_S_NREG = 112
            CONSUMER_X_NREG = 128 if store_final_correction else 112
            CONSUMER_Y_NREG = 96

            calc_mt = T.alloc_var("bool")
            if store_final_correction:
                calc_mt = True
                if use_fallback_mask:
                    calc_mt = fallback_mask[bb, bh]
            else:
                calc_mt = False

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                if use_initial_state:
                    T.copy(
                        h0[
                            bb,
                            bh,
                            0:DK,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                        h_fragment,
                    )
                else:
                    T.clear(h_fragment)

                for i_s in T.serial(num_iters):
                    if i_s > 0:
                        T.barrier_wait(iter_done, (i_s - 1) % 2)

                    left = seq_start_idx + i_s * block_S
                    right = left + block_S
                    T.copy(
                        k[batch_idx, left:right, bhg, 0:DK],
                        k_shared,
                    )
                    T.barrier_arrive(data_loaded)
                    T.barrier_wait(data_loaded, i_s % 2)

                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    T.barrier_wait(bar_1, i_s % 2)
                    g_last_local_S[0] = T.exp2(g_shared[block_S - 1] * 1.442695)
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] *= g_last_local_S[0]
                    T.barrier_arrive(bar_2)

                    T.barrier_wait(bar_2, i_s % 2)
                    T.gemm(
                        x_shared,
                        y_shared,
                        h_fragment,
                        transpose_A=True,
                        clear_accum=False,
                    )
                    if store_final_correction:
                        T.barrier_arrive(state_ready)
                    T.barrier_arrive(iter_done)

                T.copy(
                    h_fragment,
                    ht[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                )

            elif tx < 256:
                T.set_max_nreg(CONSUMER_X_NREG, 1)

                if store_final_correction:
                    if calc_mt:
                        for j_k, j_v in T.Parallel(DK, block_DV):
                            if j_k == bv * block_DV + j_v:
                                m_fragment[j_k, j_v] = 1
                            else:
                                m_fragment[j_k, j_v] = 0
                        g_prod_M[0] = 0

                for i_s in T.serial(num_iters):
                    if i_s > 0:
                        T.barrier_wait(iter_done, (i_s - 1) % 2)

                    left = seq_start_idx + i_s * block_S
                    right = left + block_S
                    T.copy(
                        a[batch_idx, left:right, bh, 0:block_S],
                        a_shared,
                    )
                    if right <= seq_end_idx:
                        for j_s in T.Parallel(block_S):
                            b_shared[j_s] = b[batch_idx, left + j_s, bh]
                    else:
                        for j_s in T.Parallel(block_S):
                            if left + j_s < seq_end_idx:
                                b_shared[j_s] = b[batch_idx, left + j_s, bh]
                            else:
                                b_shared[j_s] = 0
                    T.barrier_arrive(data_loaded)
                    T.barrier_wait(data_loaded, i_s % 2)

                    T.gemm(
                        a_shared,
                        k_shared,
                        x_fragment,
                        transpose_A=True,
                        clear_accum=True,
                    )
                    for j_s, j_k in T.Parallel(block_S, DK):
                        x_shared[j_s, j_k] = x_fragment[j_s, j_k] * -b_shared[j_s]
                    T.barrier_arrive(bar_2)

                    if store_final_correction:
                        if calc_mt:
                            T.barrier_wait(state_ready, i_s % 2)
                            g_prod_M[0] += g_shared[block_S - 1]
                            T.copy(m_fragment, m_shared)
                            T.gemm(
                                k_shared,
                                m_shared,
                                z_fragment,
                                clear_accum=True,
                            )
                            T.copy(z_fragment, z_shared)
                            T.gemm(
                                x_shared,
                                z_shared,
                                m_fragment,
                                transpose_A=True,
                                clear_accum=False,
                            )
                    T.barrier_arrive(iter_done)

                if store_final_correction:
                    if calc_mt:
                        g_prod_M[0] = T.exp2(g_prod_M[0] * 1.442695)
                        for j_k, j_v in T.Parallel(DK, block_DV):
                            m_fragment[j_k, j_v] *= g_prod_M[0]
                        T.copy(
                            m_fragment,
                            mt[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV],
                        )

            else:
                T.set_max_nreg(CONSUMER_Y_NREG, 1)

                for i_s in T.serial(num_iters):
                    if i_s > 0:
                        T.barrier_wait(iter_done, (i_s - 1) % 2)

                    left = seq_start_idx + i_s * block_S
                    right = left + block_S
                    T.copy(
                        v[
                            batch_idx,
                            left:right,
                            bh,
                            bv * block_DV : (bv + 1) * block_DV,
                        ],
                        v_shared,
                    )
                    if right <= seq_end_idx:
                        for j_s in T.Parallel(block_S):
                            g_shared[j_s] = g[batch_idx, left + j_s, bh]
                    else:
                        for j_s in T.Parallel(block_S):
                            if left + j_s < seq_end_idx:
                                g_shared[j_s] = g[batch_idx, left + j_s, bh]
                            else:
                                g_shared[j_s] = g[batch_idx, seq_end_idx - 1, bh]
                    T.barrier_arrive(data_loaded)
                    T.barrier_wait(data_loaded, i_s % 2)

                    g_last_local_Y[0] = g_shared[block_S - 1]
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.exp2(
                            (g_last_local_Y[0] - g_shared[j_s]) * 1.442695
                        )
                    g_last_local_Y[0] = T.exp2(g_last_local_Y[0] * 1.442695)
                    T.barrier_arrive(bar_1)

                    T.barrier_wait(bar_1, i_s % 2)
                    T.gemm(
                        k_shared,
                        h_shared,
                        y_fragment,
                        clear_accum=True,
                    )
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        y_shared[j_s, j_v] = (
                            y_fragment[j_s, j_v] * g_last_local_Y[0]
                            - v_shared[j_s, j_v] * g_rev_exp_shared[j_s]
                        )
                    T.barrier_arrive(bar_2)
                    T.barrier_arrive(iter_done)

    return tilelang_prepare_h_cp_v2_kernel


def fused_gdr_h(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = True,
    output_correction: bool = True,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    num_warmup_chunks: torch.LongTensor | None = None,
    fallback_mask: torch.Tensor | None = None,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    assert K == V == 128
    assert chunk_size == 64

    if cu_seqlens is None:
        assert num_warmup_chunks is None
        real_batch_size = batch_size
        num_chunks = tilelang.cdiv(num_tokens, chunk_size) if output_h else 0
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        chunk_offsets = torch.empty(
            (batch_size + 1), dtype=torch.int32, device=k.device
        )
        is_varlen = False
        is_cp = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        chunk_offsets, num_chunks = prepare_chunk_offsets(cu_seqlens, chunk_size)
        chunk_offsets = chunk_offsets.to(cu_seqlens.dtype)
        num_chunks = num_chunks if output_h else 0
        is_varlen = True
        if num_warmup_chunks is None:
            num_warmup_chunks = torch.empty(
                (real_batch_size, H), dtype=cu_seqlens.dtype, device=k.device
            )
            is_cp = False
        else:
            is_cp = True

    use_fallback_mask = fallback_mask is not None
    if fallback_mask is None:
        fallback_mask = torch.empty(
            (real_batch_size, H), dtype=torch.bool, device=k.device
        )

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )
    ht_dtype = torch.bfloat16 if is_cp else torch.float32
    final_state = torch.empty(
        (real_batch_size, H, K, V), dtype=ht_dtype, device=k.device
    )
    if is_cp and output_final_state and not output_h and not output_correction:
        block_DV = 64
        assert V % block_DV == 0
        final_correction = torch.empty(
            (real_batch_size, H, K, K), dtype=ht_dtype, device=k.device
        )
        tilelang_prepare_h_cp_kernel = tilelang_prepare_h_cp_v2(
            H,
            Hg,
            K,
            V,
            chunk_size,
            block_DV,
            qkva_dtype=k.dtype,
            g_dtype=g.dtype,
            b_dtype=b.dtype,
            h0_dtype=initial_state.dtype,
            ht_dtype=final_state.dtype,
            mask_dtype=fallback_mask.dtype,
            seqlen_dtype=cu_seqlens.dtype,
            accum_dtype="float32",
            use_initial_state=use_initial_state,
            store_final_correction=output_correction,
            use_fallback_mask=use_fallback_mask,
            is_varlen=is_varlen,
        )
        tilelang_prepare_h_cp_kernel(
            k,
            v,
            a,
            g,
            b,
            initial_state,
            cu_seqlens,
            num_warmup_chunks,
            fallback_mask,
            final_state,
            final_correction,
        )
        if not output_correction:
            final_correction = None
        return None, final_state, final_correction

    h = torch.empty((batch_size, num_chunks, H, K, V), dtype=k.dtype, device=k.device)
    final_correction = torch.empty(
        (real_batch_size, H, K, K), dtype=ht_dtype, device=k.device
    )
    num_stages = 1 if is_cp and not output_h and not output_correction else 2

    tilelang_prepare_h_kernel = tilelang_prepare_h(
        H,
        Hg,
        K,
        V,
        chunk_size,
        qkva_dtype=k.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        h_dtype=h.dtype,
        mask_dtype=fallback_mask.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        store_h=output_h,
        store_final_correction=output_correction,
        use_fallback_mask=use_fallback_mask,
        is_varlen=is_varlen,
        is_cp=is_cp,
        num_stages=num_stages,
    )
    tilelang_prepare_h_kernel(
        k,
        v,
        a,
        g,
        b,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        num_warmup_chunks,
        fallback_mask,
        h,
        final_state,
        final_correction,
    )

    if not output_final_state:
        final_state = None
        final_correction = None
    if not output_correction:
        final_correction = None
    if not output_h:
        h = None

    return h, final_state, final_correction
