# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os

import torch
import tilelang
import tilelang.language as T


def _debug_enabled() -> bool:
    return os.environ.get("FLASHQLA_DEBUG_BLACKWELL_DISPATCH", "") == "1"


_DEBUG_MESSAGES = set()


def _debug(message: str):
    if _debug_enabled():
        if os.environ.get("FLASHQLA_DEBUG_BLACKWELL_DISPATCH_REPEAT", "") != "1":
            if message in _DEBUG_MESSAGES:
                return
            _DEBUG_MESSAGES.add(message)
        print(f"[FlashQLA Blackwell fwd native] {message}", flush=True)


def _select_block_dv(real_batch_size: int, num_v_heads: int) -> int:
    try:
        sm_count = torch.cuda.get_device_properties().multi_processor_count
    except Exception:
        sm_count = 148
    ratio = 0.7
    target_num_ctas = max(1, int(sm_count * ratio))
    grid_size = real_batch_size * num_v_heads
    if grid_size >= target_num_ctas:
        return 128
    if grid_size * 2 >= target_num_ctas:
        return 64
    return 64


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_gdr_fwd_blackwell_ag(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    store_o,
    is_varlen,
    max_iters,
    use_bar_load,
    use_bar_h_shared,
    use_bar_o,
    use_bar_h_scaled,
    is_cp,
    num_threads=128,
    block_DV=64,
    tmem_width=128,
):
    batch_size = T.dynamic("batch_size")
    raw_batch_size = T.dynamic("raw_batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        o_shape = (1, num_tokens, H, DV)
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        o_shape = (batch_size, num_tokens, H, DV)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (raw_batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_blackwell_ag_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        cp_seq_map: T.Tensor([batch_size], dtype=seqlen_dtype),
        raw_cu_seqlens: T.Tensor([raw_batch_size + 1], dtype=seqlen_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=num_threads) as (
            bbhv,
        ):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)
            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            raw_batch_idx = T.alloc_var("int32")
            raw_seq_end_idx = T.alloc_var("int32")
            need_store_final_state = T.alloc_var("bool")
            raw_batch_idx = cp_seq_map[bb] if is_cp else bb
            raw_seq_end_idx = (
                raw_cu_seqlens[raw_batch_idx + 1] if is_cp else seq_end_idx
            )
            need_store_final_state = store_final_state & (
                raw_seq_end_idx == seq_end_idx
            )

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_pad_shared = T.alloc_shared((tmem_width, block_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            g_inv_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            pg_fragment = T.alloc_fragment((block_S, tmem_width), dtype=qkva_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            # Keep TMEM allocations on TCGEN05-legal widths; some 64-wide
            # logical tiles are padded to the 128-wide Blackwell MMA atom.
            h_tmem = T.alloc_tmem((DK, tmem_width), dtype=accum_dtype)
            tmp_tmem = T.alloc_tmem((block_S, tmem_width), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            pg_tmem = T.alloc_tmem((block_S, tmem_width), dtype=qkva_dtype)

            mbar_u = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_v = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_p = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o0 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o1 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_h = T.alloc_barrier(arrive_count=[1] * 8)
            bar_load = T.alloc_barrier(arrive_count=num_threads)
            bar_h_shared = T.alloc_barrier(arrive_count=num_threads)
            bar_o = T.alloc_barrier(arrive_count=num_threads)
            bar_h_scaled = T.alloc_barrier(arrive_count=num_threads)

            T.use_swizzle(10)

            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S)
            if max_iters > 0 and num_iters > max_iters:
                num_iters = max_iters

            if use_initial_state:
                T.copy(h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], h_fragment)
            else:
                T.clear(h_fragment)

            for i_s in T.serial(num_iters):
                left = seq_start_idx + i_s * block_S
                right = left + block_S
                mbar_slot = i_s % 8
                mbar_phase = (i_s // 8) % 2

                if right <= seq_end_idx:
                    T.copy(q[batch_idx, left:right, bhg, 0:DK], q_shared)
                    T.copy(k[batch_idx, left:right, bhg, 0:DK], k_shared)
                    T.copy(v[batch_idx, left:right, bh, bv * block_DV : (bv + 1) * block_DV], v_shared)
                    T.copy(a[batch_idx, left:right, bh, 0:block_S], a_shared)
                    for j_s in T.Parallel(block_S):
                        g_shared[j_s] = g[batch_idx, left + j_s, bh]
                    for j_s in T.Parallel(block_S):
                        b_shared[j_s] = b[batch_idx, left + j_s, bh]
                else:
                    for j_s, j_k in T.Parallel(block_S, DK):
                        if left + j_s < seq_end_idx:
                            q_shared[j_s, j_k] = q[batch_idx, left + j_s, bhg, j_k]
                            k_shared[j_s, j_k] = k[batch_idx, left + j_s, bhg, j_k]
                        else:
                            q_shared[j_s, j_k] = 0
                            k_shared[j_s, j_k] = 0
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        if left + j_s < seq_end_idx:
                            v_shared[j_s, j_v] = v[
                                batch_idx,
                                left + j_s,
                                bh,
                                bv * block_DV + j_v,
                            ]
                        else:
                            v_shared[j_s, j_v] = 0
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if left + j_s < seq_end_idx:
                            if left + j_t < seq_end_idx:
                                a_shared[j_s, j_t] = a[batch_idx, left + j_s, bh, j_t]
                            else:
                                a_shared[j_s, j_t] = 0
                        else:
                            a_shared[j_s, j_t] = 0
                    for j_s in T.Parallel(block_S):
                        if left + j_s < seq_end_idx:
                            g_shared[j_s] = g[batch_idx, left + j_s, bh]
                        else:
                            g_shared[j_s] = g[batch_idx, seq_end_idx - 1, bh]
                    for j_s in T.Parallel(block_S):
                        if left + j_s < seq_end_idx:
                            b_shared[j_s] = b[batch_idx, left + j_s, bh]
                        else:
                            b_shared[j_s] = 0

                for j_s in T.Parallel(block_S):
                    g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                    g_inv_exp_shared[j_s] = 1.0 / g_exp_shared[j_s]
                if use_bar_load:
                    T.barrier_arrive(bar_load)
                    T.barrier_wait(bar_load, i_s % 2)
                for j_s in T.Parallel(block_S):
                    g_rev_exp_shared[j_s] = T.if_then_else(
                        left + j_s < seq_end_idx,
                        g_exp_shared[block_S - 1] * g_inv_exp_shared[j_s],
                        0.0,
                    )
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s >= j_t:
                        a_shared[j_s, j_t] *= (
                            g_exp_shared[j_s] * g_inv_exp_shared[j_t]
                        )
                        a_shared[j_s, j_t] *= b_shared[j_t]
                    else:
                        a_shared[j_s, j_t] = 0

                # h_shared holds the previous recurrent state for this chunk.
                T.copy(h_fragment, h_shared)
                if use_bar_h_shared:
                    T.barrier_arrive(bar_h_shared)
                    T.barrier_wait(bar_h_shared, i_s % 2)

                # U = K @ S
                T.tcgen05_gemm(
                    k_shared,
                    h_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=True,
                    mbar=mbar_u[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_u[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], u_fragment)

                # W = V - g * U
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                    v_shared[j_s, j_v] = u_fragment[j_s, j_v]

                # Vd = Ag @ W
                T.tcgen05_gemm(
                    a_shared,
                    v_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=True,
                    mbar=mbar_v[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_v[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], v_fragment)
                for j_s, j_v in T.Parallel(tmem_width, block_DV):
                    if j_s < block_S:
                        vd_pad_shared[j_s, j_v] = v_fragment[j_s, j_v]
                    else:
                        vd_pad_shared[j_s, j_v] = 0

                # V' = g_last / g * Vd
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    vn_shared[j_s, j_v] = v_fragment[j_s, j_v]

                # P = Q @ K^T.
                T.tcgen05_gemm(
                    q_shared,
                    k_shared,
                    p_tmem,
                    transpose_B=True,
                    clear_accum=True,
                    mbar=mbar_p[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_p[mbar_slot], mbar_phase)
                T.copy(p_tmem, p_fragment)

                # O = Q @ S
                T.tcgen05_gemm(
                    q_shared,
                    h_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=True,
                    mbar=mbar_o0[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_o0[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], o_fragment)

                # Pg = scale * G * P; O = scale * g * O + Pg @ Vd
                for j_s, j_t in T.Parallel(block_S, tmem_width):
                    if j_t < block_S:
                        if j_s >= j_t:
                            pg_fragment[j_s, j_t] = p_fragment[j_s, j_t] * (
                                scale * g_exp_shared[j_s] * g_inv_exp_shared[j_t]
                            )
                        else:
                            pg_fragment[j_s, j_t] = 0
                    else:
                        pg_fragment[j_s, j_t] = 0
                T.copy(pg_fragment, pg_tmem)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.copy(o_fragment, tmp_tmem[:, 0:block_DV])
                if use_bar_o:
                    T.barrier_arrive(bar_o)
                    T.barrier_wait(bar_o, i_s % 2)
                T.tcgen05_gemm(
                    pg_tmem,
                    vd_pad_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=False,
                    mbar=mbar_o1[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_o1[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], o_fragment)

                if store_o:
                    if right <= seq_end_idx:
                        T.copy(o_fragment, o[batch_idx, left:right, bh, bv * block_DV : (bv + 1) * block_DV])
                    else:
                        for j_s, j_v in T.Parallel(block_S, block_DV):
                            if left + j_s < seq_end_idx:
                                o[
                                    batch_idx,
                                    left + j_s,
                                    bh,
                                    bv * block_DV + j_v,
                                ] = o_fragment[j_s, j_v]

                # S = g_last * S + K^T @ V'
                g_last_local[0] = g_exp_shared[block_S - 1]
                for j_k, j_v in T.Parallel(DK, block_DV):
                    h_fragment[j_k, j_v] *= g_last_local[0]
                T.copy(h_fragment, h_tmem[:, 0:block_DV])
                if use_bar_h_scaled:
                    T.barrier_arrive(bar_h_scaled)
                    T.barrier_wait(bar_h_scaled, i_s % 2)
                T.tcgen05_gemm(
                    k_shared,
                    vn_shared,
                    h_tmem[:, 0:block_DV],
                    transpose_A=True,
                    clear_accum=False,
                    mbar=mbar_h[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_h[mbar_slot], mbar_phase)
                T.copy(h_tmem[:, 0:block_DV], h_fragment)

            if need_store_final_state:
                T.copy(h_fragment, ht[raw_batch_idx, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return tilelang_fused_chunk_gdr_fwd_blackwell_ag_kernel


def fused_gdr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    output_o: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cp_seq_map: torch.LongTensor | None = None,
    raw_cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)

    unsupported_reasons = []
    if output_h:
        unsupported_reasons.append("output_h")
    if cu_seqlens is not None:
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        if bool((seqlens <= 0).any().item()):
            unsupported_reasons.append("varlen_empty")

    if unsupported_reasons:
        reason = ",".join(unsupported_reasons)
        _debug("unsupported reason=" + reason)
        raise NotImplementedError(
            "Blackwell native fused_gdr_fwd does not support this invocation: "
            f"{reason}. Hopper fallback is disabled on Blackwell."
        )

    _debug(
        f"using native fwd H={H} Hg={Hg} tokens={num_tokens} "
        f"varlen={cu_seqlens is not None} "
        f"output_final_state={output_final_state} output_o={output_o}"
    )
    assert K == V == 128
    assert chunk_size == 64

    is_varlen = cu_seqlens is not None
    is_cp = cp_seq_map is not None
    if is_varlen:
        real_batch_size = len(cu_seqlens) - 1
        seqlen_dtype = cu_seqlens.dtype
        if int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != num_tokens:
            raise ValueError(
                "cu_seqlens must start at 0 and end at the flattened token count "
                f"{num_tokens}, got start={int(cu_seqlens[0].item())} "
                f"end={int(cu_seqlens[-1].item())}."
            )
    else:
        real_batch_size = batch_size
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        seqlen_dtype = torch.int32
    if cp_seq_map is None:
        cp_seq_map = torch.empty(
            (real_batch_size,), dtype=seqlen_dtype, device=k.device
        )
    if raw_cu_seqlens is None:
        raw_batch_size = real_batch_size
        raw_cu_seqlens = torch.empty(
            (raw_batch_size + 1,), dtype=seqlen_dtype, device=k.device
        )
    else:
        raw_batch_size = raw_cu_seqlens.shape[0] - 1
    use_initial_state = initial_state is not None
    if initial_state is not None and initial_state.shape[0] != real_batch_size:
        raise ValueError(
            "initial_state batch dimension must match the active sequence batch "
            f"for Blackwell native fwd, expected {real_batch_size}, got "
            f"{initial_state.shape[0]}."
        )
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )

    final_state = torch.empty(
        (raw_batch_size, H, K, V), dtype=torch.float32, device=k.device
    )
    h = torch.empty((batch_size, 0, H, K, V), dtype=k.dtype, device=k.device)
    o = torch.empty_like(v)

    block_DV = _select_block_dv(real_batch_size, H)
    if block_DV not in (32, 64, 128):
        raise ValueError(
            f"Blackwell native fwd selected invalid block_DV={block_DV}"
        )
    tmem_width = 128
    max_iters = 0
    num_threads = 256
    if cu_seqlens is None:
        has_ragged_tail = num_tokens % chunk_size != 0
    else:
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        has_ragged_tail = bool((seqlens % chunk_size != 0).any().item())
    _debug(
        f"threads={num_threads} block_DV={block_DV} tmem_width={tmem_width} "
        f"ragged_tail={has_ragged_tail}"
    )
    tilelang_fused_chunk_gdr_fwd_kernel = tilelang_fused_chunk_gdr_fwd_blackwell_ag(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        store_o=output_o,
        is_varlen=is_varlen,
        max_iters=max_iters,
        use_bar_load=True,
        use_bar_h_shared=True,
        use_bar_o=True,
        use_bar_h_scaled=has_ragged_tail,
        is_cp=is_cp,
        num_threads=num_threads,
        block_DV=block_DV,
        tmem_width=tmem_width,
    )
    tilelang_fused_chunk_gdr_fwd_kernel(
        q,
        k,
        v,
        a,
        g,
        b,
        initial_state,
        cu_seqlens,
        cp_seq_map,
        raw_cu_seqlens,
        o,
        final_state,
    )

    if not output_final_state:
        final_state = None
    if not output_o:
        o = None
    return o, h, final_state
