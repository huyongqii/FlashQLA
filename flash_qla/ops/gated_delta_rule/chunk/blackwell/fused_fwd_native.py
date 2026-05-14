# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os

import torch
import tilelang
import tilelang.language as T

from flash_qla.ops.gated_delta_rule.chunk.hopper.fused_fwd import (
    fused_gdr_fwd as hopper_fused_gdr_fwd,
)


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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_gdr_fwd_blackwell_native(
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
    use_initial_state,
    store_final_state,
    store_o,
    max_iters,
    block_DV=64,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    g_shape = (batch_size, num_tokens, H)
    b_shape = (batch_size, num_tokens, H)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_blackwell_native_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=128) as (
            bbhv,
        ):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            # Keep TMEM allocations small and reusable. This first native kernel
            # is sequential by design; later versions should keep S/O live in
            # TMEM and add producer/consumer pipelining.
            h_tmem = T.alloc_tmem((DK, 128), dtype=accum_dtype)
            tmp_tmem = T.alloc_tmem((block_S, 128), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)

            mbar_u = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_v = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_p = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o0 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o1 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_h = T.alloc_barrier(arrive_count=[1] * 8)
            bar_load = T.alloc_barrier(arrive_count=128)
            bar_h_shared = T.alloc_barrier(arrive_count=128)
            bar_ag = T.alloc_barrier(arrive_count=128)
            bar_o = T.alloc_barrier(arrive_count=128)
            bar_h_scaled = T.alloc_barrier(arrive_count=128)

            num_iters = T.ceildiv(num_tokens, block_S)
            if max_iters > 0 and num_iters > max_iters:
                num_iters = max_iters

            if use_initial_state:
                T.copy(h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], h_fragment)
            else:
                T.clear(h_fragment)

            for i_s in T.serial(num_iters):
                left = i_s * block_S
                right = left + block_S
                mbar_slot = i_s % 8
                mbar_phase = (i_s // 8) % 2

                T.copy(q[bb, left:right, bhg, 0:DK], q_shared)
                T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
                T.copy(v[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV], v_shared)
                T.copy(a[bb, left:right, bh, 0:block_S], a_shared)
                for j_s in T.Parallel(block_S):
                    g_shared[j_s] = g[bb, left + j_s, bh]
                    b_shared[j_s] = b[bb, left + j_s, bh]

                for j_s in T.Parallel(block_S):
                    g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                    g_rev_exp_shared[j_s] = T.exp2(
                        (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                    )
                T.barrier_arrive(bar_load)
                T.barrier_wait(bar_load, i_s % 2)

                # h_shared holds the previous recurrent state for this chunk.
                T.copy(h_fragment, h_shared)
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

                # Ag = G * A * beta
                for j_s, j_t in T.Parallel(block_S, block_S):
                    g_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s >= j_t:
                        g_fragment[j_s, j_t] = T.exp2(
                            g_fragment[j_s, j_t] * 1.442695
                        )
                    else:
                        g_fragment[j_s, j_t] = 0
                for j_s, j_t in T.Parallel(block_S, block_S):
                    a_fragment[j_s, j_t] = a_shared[j_s, j_t]
                    a_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                    a_fragment[j_s, j_t] *= b_shared[j_t]
                    a_shared[j_s, j_t] = a_fragment[j_s, j_t]
                T.barrier_arrive(bar_ag)
                T.barrier_wait(bar_ag, i_s % 2)

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
                T.copy(v_fragment, vd_shared)

                # V' = g_last / g * Vd
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    vn_shared[j_s, j_v] = v_fragment[j_s, j_v]

                # P = Q @ K^T
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
                for j_s, j_t in T.Parallel(block_S, block_S):
                    p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    p_shared[j_s, j_t] = p_fragment[j_s, j_t]
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.copy(o_fragment, tmp_tmem[:, 0:block_DV])
                T.barrier_arrive(bar_o)
                T.barrier_wait(bar_o, i_s % 2)
                T.tcgen05_gemm(
                    p_shared,
                    vd_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=False,
                    mbar=mbar_o1[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_o1[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], o_fragment)

                if store_o:
                    T.copy(o_fragment, o[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV])

                # S = g_last * S + K^T @ V'
                g_last_local[0] = g_exp_shared[block_S - 1]
                for j_k, j_v in T.Parallel(DK, block_DV):
                    h_fragment[j_k, j_v] *= g_last_local[0]
                T.copy(h_fragment, h_tmem[:, 0:block_DV])
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

            if store_final_state:
                T.copy(h_fragment, ht[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return tilelang_fused_chunk_gdr_fwd_blackwell_native_kernel


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

    fallback_reasons = []
    if os.environ.get("FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE", "") != "1":
        fallback_reasons.append("native_fwd_disabled")
    if output_h:
        fallback_reasons.append("output_h")
    if cu_seqlens is not None:
        fallback_reasons.append("varlen")
    if cp_seq_map is not None:
        fallback_reasons.append("cp_seq_map")
    if raw_cu_seqlens is not None:
        fallback_reasons.append("raw_cu_seqlens")
    if num_tokens % chunk_size != 0:
        fallback_reasons.append("ragged_tokens")

    if fallback_reasons:
        _debug("fallback=hopper reason=" + ",".join(fallback_reasons))
        return hopper_fused_gdr_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            g=g,
            b=b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_h=output_h,
            output_o=output_o,
            cu_seqlens=cu_seqlens,
            cp_seq_map=cp_seq_map,
            raw_cu_seqlens=raw_cu_seqlens,
            chunk_size=chunk_size,
        )

    _debug(
        f"using native fixed-length fwd H={H} Hg={Hg} tokens={num_tokens} "
        f"output_final_state={output_final_state} output_o={output_o}"
    )
    assert K == V == 128
    assert chunk_size == 64

    real_batch_size = batch_size
    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )

    final_state = torch.empty(
        (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
    )
    h = torch.empty((batch_size, 0, H, K, V), dtype=k.dtype, device=k.device)
    o = torch.empty_like(v)

    block_DV = int(os.environ.get("FLASHQLA_BLACKWELL_BLOCK_DV", "64"))
    if block_DV not in (64, 128):
        raise ValueError(
            "FLASHQLA_BLACKWELL_BLOCK_DV must be 64 or 128 for the current "
            f"TileLang 0.1.9 TCGEN05 path, got {block_DV}"
        )
    max_iters = int(os.environ.get("FLASHQLA_BLACKWELL_FWD_MAX_ITERS", "0"))
    if max_iters > 0:
        _debug(f"debug max_iters={max_iters}; output is partial and benchmark is invalid")
    tilelang_fused_chunk_gdr_fwd_kernel = tilelang_fused_chunk_gdr_fwd_blackwell_native(
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
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        store_o=output_o,
        max_iters=max_iters,
        block_DV=block_DV,
    )
    tilelang_fused_chunk_gdr_fwd_kernel(
        q,
        k,
        v,
        a,
        g,
        b,
        initial_state,
        o,
        final_state,
    )

    if not output_final_state:
        final_state = None
    if not output_o:
        o = None
    return o, h, final_state
