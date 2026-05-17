# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os

import torch
import tilelang
import tilelang.language as T

from flash_qla.ops.gated_delta_rule.chunk.hopper.fused_fwd import (
    fused_gdr_fwd as hopper_fused_gdr_fwd,
)
from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_h import (
    fused_gdr_h as hopper_fused_gdr_h,
)
from flash_qla.ops.gated_delta_rule.chunk.blackwell.policy import should_use_native_fwd


def _debug_enabled() -> bool:
    return os.environ.get("FLASHQLA_DEBUG_BLACKWELL_DISPATCH", "") == "1"


_DEBUG_MESSAGES = set()
_DEFAULT_SYNC_BARRIERS = frozenset(("load", "h"))
_VALID_SYNC_BARRIERS = frozenset(("load", "h", "ag", "o", "hscale"))


def _debug(message: str):
    if _debug_enabled():
        if os.environ.get("FLASHQLA_DEBUG_BLACKWELL_DISPATCH_REPEAT", "") != "1":
            if message in _DEBUG_MESSAGES:
                return
            _DEBUG_MESSAGES.add(message)
        print(f"[FlashQLA Blackwell fwd native] {message}", flush=True)


def _sync_barriers() -> set[str]:
    value = os.environ.get("FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS")
    if value is None:
        return set(_DEFAULT_SYNC_BARRIERS)
    value = value.strip().lower()
    if value in ("", "none", "0", "off"):
        return set()
    if value in ("default", "safe", "all", "1", "on"):
        return set(_DEFAULT_SYNC_BARRIERS)
    aliases = {
        "h_shared": "h",
        "hshared": "h",
        "h_scaled": "hscale",
        "hscaled": "hscale",
    }
    barriers = set()
    for item in value.split(","):
        item = aliases.get(item.strip(), item.strip())
        if item:
            barriers.add(item)
    unknown = barriers - _VALID_SYNC_BARRIERS
    if unknown:
        raise ValueError(
            "Unknown FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS entries: "
            f"{sorted(unknown)}. Valid entries are {sorted(_VALID_SYNC_BARRIERS)}"
        )
    return barriers


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
    h0_dtype,
    ht_dtype,
    o_dtype,
    use_initial_state,
    store_final_state,
    store_o,
    max_iters,
    use_bar_load,
    use_bar_h_shared,
    use_bar_o,
    use_bar_h_scaled,
    num_threads=128,
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
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_blackwell_ag_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=num_threads) as (
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
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
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
            g_last_local = T.alloc_local((1), dtype=accum_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

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
            bar_load = T.alloc_barrier(arrive_count=num_threads)
            bar_h_shared = T.alloc_barrier(arrive_count=num_threads)
            bar_o = T.alloc_barrier(arrive_count=num_threads)
            bar_h_scaled = T.alloc_barrier(arrive_count=num_threads)

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

                for j_s in T.Parallel(block_S):
                    g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                    g_rev_exp_shared[j_s] = T.exp2(
                        (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                    )
                if use_bar_load:
                    T.barrier_arrive(bar_load)
                    T.barrier_wait(bar_load, i_s % 2)

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

                # The fixed-length Blackwell fast KKT path precomputes Ag =
                # G * A * beta, so this kernel only needs G for the P path.
                for j_s, j_t in T.Parallel(block_S, block_S):
                    g_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s >= j_t:
                        g_fragment[j_s, j_t] = T.exp2(
                            g_fragment[j_s, j_t] * 1.442695
                        )
                    else:
                        g_fragment[j_s, j_t] = 0

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
                T.copy(p_fragment, p_shared)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.copy(o_fragment, tmp_tmem[:, 0:block_DV])
                if use_bar_o:
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

            if store_final_state:
                T.copy(h_fragment, ht[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return tilelang_fused_chunk_gdr_fwd_blackwell_ag_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_precompute_pg_blackwell(
    H,
    Hg,
    DK,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    pg_dtype=None,
):
    pg_dtype = pg_dtype or qkva_dtype
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    g_shape = (batch_size, num_tokens, H)
    pg_shape = (batch_size, num_tokens, H, chunk_size)

    @T.prim_func
    def tilelang_precompute_pg_blackwell_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        pg: T.Tensor(pg_shape, dtype=pg_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(num_chunks * H, threads=128) as (bch,):
            bc, bh = bch // H, bch % H
            bb = bc % batch_size
            chunk_idx = bc // batch_size
            bhg = bh // (H // Hg)
            left = chunk_idx * block_S
            right = left + block_S

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            mbar_p = T.alloc_barrier(arrive_count=1)
            bar_load = T.alloc_barrier(arrive_count=128)

            T.copy(q[bb, left:right, bhg, 0:DK], q_shared)
            T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
            for j_s in T.Parallel(block_S):
                g_shared[j_s] = g[bb, left + j_s, bh]
            T.barrier_arrive(bar_load)
            T.barrier_wait(bar_load, 0)

            T.tcgen05_gemm(
                q_shared,
                k_shared,
                p_tmem,
                transpose_B=True,
                clear_accum=True,
                mbar=mbar_p,
            )
            T.mbarrier_wait_parity(mbar_p, 0)
            T.copy(p_tmem, p_fragment)

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
                p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
            for j_s, j_t in T.Parallel(block_S, block_S):
                pg[bb, left + j_s, bh, j_t] = p_fragment[j_s, j_t]

    return tilelang_precompute_pg_blackwell_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_precompute_p_blackwell(
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    p_shape = (batch_size, num_tokens, Hg, chunk_size)

    @T.prim_func
    def tilelang_precompute_p_blackwell_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        p: T.Tensor(p_shape, dtype=accum_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(num_chunks * Hg, threads=128) as (bcg,):
            bc, bhg = bcg // Hg, bcg % Hg
            bb = bc % batch_size
            chunk_idx = bc // batch_size
            left = chunk_idx * block_S
            right = left + block_S

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)
            mbar_p = T.alloc_barrier(arrive_count=1)
            bar_load = T.alloc_barrier(arrive_count=128)

            T.copy(q[bb, left:right, bhg, 0:DK], q_shared)
            T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
            T.barrier_arrive(bar_load)
            T.barrier_wait(bar_load, 0)

            T.tcgen05_gemm(
                q_shared,
                k_shared,
                p_tmem,
                transpose_B=True,
                clear_accum=True,
                mbar=mbar_p,
            )
            T.mbarrier_wait_parity(mbar_p, 0)
            T.copy(p_tmem, p_fragment)
            T.copy(p_fragment, p_shared)
            for j_s, j_t in T.Parallel(block_S, block_S):
                p[bb, left + j_s, bhg, j_t] = p_shared[j_s, j_t]

    return tilelang_precompute_p_blackwell_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_gdr_fwd_blackwell_pg_input(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    use_initial_state,
    store_final_state,
    store_o,
    max_iters,
    pg_dtype=None,
    num_threads=256,
    block_DV=64,
):
    pg_dtype = pg_dtype or qkva_dtype
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    g_shape = (batch_size, num_tokens, H)
    pg_shape = (batch_size, num_tokens, H, chunk_size)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_blackwell_pg_input_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        pg: T.Tensor(pg_shape, dtype=pg_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=num_threads) as (
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
            pg_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
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
            pg_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            h_tmem = T.alloc_tmem((DK, 128), dtype=accum_dtype)
            tmp_tmem = T.alloc_tmem((block_S, 128), dtype=accum_dtype)

            mbar_u = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_v = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o0 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o1 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_h = T.alloc_barrier(arrive_count=[1] * 8)
            bar_load = T.alloc_barrier(arrive_count=num_threads)
            bar_h_shared = T.alloc_barrier(arrive_count=num_threads)
            bar_pg_shared = T.alloc_barrier(arrive_count=num_threads)

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
                for j_s in T.Parallel(block_S):
                    g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                    g_rev_exp_shared[j_s] = T.exp2(
                        (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                    )
                T.barrier_arrive(bar_load)
                T.barrier_wait(bar_load, i_s % 2)

                T.copy(h_fragment, h_shared)
                T.barrier_arrive(bar_h_shared)
                T.barrier_wait(bar_h_shared, i_s % 2)

                T.tcgen05_gemm(
                    k_shared,
                    h_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=True,
                    mbar=mbar_u[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_u[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], u_fragment)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                    v_shared[j_s, j_v] = u_fragment[j_s, j_v]

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
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    vn_shared[j_s, j_v] = v_fragment[j_s, j_v]

                for j_s, j_t in T.Parallel(block_S, block_S):
                    pg_fragment[j_s, j_t] = pg[bb, left + j_s, bh, j_t]
                T.copy(pg_fragment, pg_shared)
                T.barrier_arrive(bar_pg_shared)
                T.barrier_wait(bar_pg_shared, i_s % 2)

                T.tcgen05_gemm(
                    q_shared,
                    h_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=True,
                    mbar=mbar_o0[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_o0[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], o_fragment)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.copy(o_fragment, tmp_tmem[:, 0:block_DV])
                T.tcgen05_gemm(
                    pg_shared,
                    vd_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=False,
                    mbar=mbar_o1[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_o1[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], o_fragment)
                if store_o:
                    T.copy(o_fragment, o[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV])

                g_last_local[0] = g_exp_shared[block_S - 1]
                for j_k, j_v in T.Parallel(DK, block_DV):
                    h_fragment[j_k, j_v] *= g_last_local[0]
                T.copy(h_fragment, h_tmem[:, 0:block_DV])
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

    return tilelang_fused_chunk_gdr_fwd_blackwell_pg_input_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_chunk_gdr_o_from_h_blackwell(
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
    h_dtype,
    o_dtype,
    block_DV=64,
    num_threads=256,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    g_shape = (batch_size, num_tokens, H)
    b_shape = (batch_size, num_tokens, H)
    h_shape = (batch_size, num_chunks, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_chunk_gdr_o_from_h_blackwell_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
    ):
        with T.Kernel(
            batch_size * num_chunks * H * T.ceildiv(DV, block_DV),
            threads=num_threads,
        ) as (bid,):
            bchv, bv = bid // T.ceildiv(DV, block_DV), bid % T.ceildiv(DV, block_DV)
            bch, bh = bchv // H, bchv % H
            bb, bc = bch // num_chunks, bch % num_chunks
            bhg = bh // (H // Hg)
            left = bc * block_S
            right = left + block_S

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            a_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            tmp_tmem = T.alloc_tmem((block_S, 128), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)

            mbar_u = T.alloc_barrier(arrive_count=1)
            mbar_v = T.alloc_barrier(arrive_count=1)
            mbar_p = T.alloc_barrier(arrive_count=1)
            mbar_o0 = T.alloc_barrier(arrive_count=1)
            mbar_o1 = T.alloc_barrier(arrive_count=1)
            bar_load = T.alloc_barrier(arrive_count=num_threads)
            bar_p_shared = T.alloc_barrier(arrive_count=num_threads)

            T.copy(q[bb, left:right, bhg, 0:DK], q_shared)
            T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
            T.copy(v[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV], v_shared)
            T.copy(a[bb, left:right, bh, 0:block_S], a_shared)
            T.copy(h[bb, bc, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], h_shared)
            for j_s in T.Parallel(block_S):
                g_shared[j_s] = g[bb, left + j_s, bh]
                b_shared[j_s] = b[bb, left + j_s, bh]
            for j_s in T.Parallel(block_S):
                g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                g_rev_exp_shared[j_s] = T.exp2(
                    (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                )
            T.barrier_arrive(bar_load)
            T.barrier_wait(bar_load, 0)

            T.tcgen05_gemm(
                k_shared,
                h_shared,
                tmp_tmem[:, 0:block_DV],
                clear_accum=True,
                mbar=mbar_u,
            )
            T.mbarrier_wait_parity(mbar_u, 0)
            T.copy(tmp_tmem[:, 0:block_DV], u_fragment)
            for j_s, j_v in T.Parallel(block_S, block_DV):
                u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                v_shared[j_s, j_v] = u_fragment[j_s, j_v]

            for j_s, j_t in T.Parallel(block_S, block_S):
                g_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
            for j_s, j_t in T.Parallel(block_S, block_S):
                if j_s >= j_t:
                    g_fragment[j_s, j_t] = T.exp2(g_fragment[j_s, j_t] * 1.442695)
                else:
                    g_fragment[j_s, j_t] = 0
            for j_s, j_t in T.Parallel(block_S, block_S):
                a_fragment[j_s, j_t] = a_shared[j_s, j_t] * g_fragment[j_s, j_t]
            for j_s, j_t in T.Parallel(block_S, block_S):
                a_fragment[j_s, j_t] *= b_shared[j_t]
            T.copy(a_fragment, a_shared)

            T.tcgen05_gemm(
                a_shared,
                v_shared,
                tmp_tmem[:, 0:block_DV],
                clear_accum=True,
                mbar=mbar_v,
            )
            T.mbarrier_wait_parity(mbar_v, 0)
            T.copy(tmp_tmem[:, 0:block_DV], v_fragment)
            T.copy(v_fragment, vd_shared)

            T.tcgen05_gemm(
                q_shared,
                k_shared,
                p_tmem,
                transpose_B=True,
                clear_accum=True,
                mbar=mbar_p,
            )
            T.mbarrier_wait_parity(mbar_p, 0)
            T.copy(p_tmem, p_fragment)
            for j_s, j_t in T.Parallel(block_S, block_S):
                p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
            T.copy(p_fragment, p_shared)
            T.barrier_arrive(bar_p_shared)
            T.barrier_wait(bar_p_shared, 0)

            T.tcgen05_gemm(
                q_shared,
                h_shared,
                tmp_tmem[:, 0:block_DV],
                clear_accum=True,
                mbar=mbar_o0,
            )
            T.mbarrier_wait_parity(mbar_o0, 0)
            T.copy(tmp_tmem[:, 0:block_DV], o_fragment)
            for j_s, j_v in T.Parallel(block_S, block_DV):
                o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
            T.copy(o_fragment, tmp_tmem[:, 0:block_DV])
            T.tcgen05_gemm(
                p_shared,
                vd_shared,
                tmp_tmem[:, 0:block_DV],
                clear_accum=False,
                mbar=mbar_o1,
            )
            T.mbarrier_wait_parity(mbar_o1, 0)
            T.copy(tmp_tmem[:, 0:block_DV], o_fragment)
            T.copy(o_fragment, o[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV])

    return tilelang_chunk_gdr_o_from_h_blackwell_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_gdr_fwd_blackwell_p_input(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    use_initial_state,
    store_final_state,
    store_o,
    max_iters,
    recompute_p_for_debug,
    num_threads=256,
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
    p_shape = (batch_size, num_tokens, Hg, chunk_size)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_blackwell_p_input_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        p: T.Tensor(p_shape, dtype=accum_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=num_threads) as (
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
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            h_tmem = T.alloc_tmem((DK, 128), dtype=accum_dtype)
            tmp_tmem = T.alloc_tmem((block_S, 128), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)

            mbar_u = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_v = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o0 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o1 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_h = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_p = T.alloc_barrier(arrive_count=[1] * 8)
            bar_load = T.alloc_barrier(arrive_count=num_threads)
            bar_h_shared = T.alloc_barrier(arrive_count=num_threads)
            bar_p_shared = T.alloc_barrier(arrive_count=num_threads)

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
                for j_s in T.Parallel(block_S):
                    g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                    g_rev_exp_shared[j_s] = T.exp2(
                        (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                    )
                T.barrier_arrive(bar_load)
                T.barrier_wait(bar_load, i_s % 2)

                T.copy(h_fragment, h_shared)
                T.barrier_arrive(bar_h_shared)
                T.barrier_wait(bar_h_shared, i_s % 2)

                T.tcgen05_gemm(
                    k_shared,
                    h_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=True,
                    mbar=mbar_u[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_u[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], u_fragment)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                    v_shared[j_s, j_v] = u_fragment[j_s, j_v]

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
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    vn_shared[j_s, j_v] = v_fragment[j_s, j_v]

                T.tcgen05_gemm(
                    q_shared,
                    h_shared,
                    tmp_tmem[:, 0:block_DV],
                    clear_accum=True,
                    mbar=mbar_o0[mbar_slot],
                )
                T.mbarrier_wait_parity(mbar_o0[mbar_slot], mbar_phase)
                T.copy(tmp_tmem[:, 0:block_DV], o_fragment)

                if recompute_p_for_debug:
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
                for j_s, j_t in T.Parallel(block_S, block_S):
                    g_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s >= j_t:
                        g_fragment[j_s, j_t] = T.exp2(
                            g_fragment[j_s, j_t] * 1.442695
                        )
                    else:
                        g_fragment[j_s, j_t] = 0
                if recompute_p_for_debug:
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    T.copy(p_fragment, p_shared)
                else:
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] = p[bb, left + j_s, bhg, j_t]
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    T.copy(p_fragment, p_shared)
                T.barrier_arrive(bar_p_shared)
                T.barrier_wait(bar_p_shared, i_s % 2)
                for j_s, j_v in T.Parallel(block_S, block_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.copy(o_fragment, tmp_tmem[:, 0:block_DV])
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

                g_last_local[0] = g_exp_shared[block_S - 1]
                for j_k, j_v in T.Parallel(DK, block_DV):
                    h_fragment[j_k, j_v] *= g_last_local[0]
                T.copy(h_fragment, h_tmem[:, 0:block_DV])
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

    return tilelang_fused_chunk_gdr_fwd_blackwell_p_input_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_gdr_fwd_blackwell_pipeline(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
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
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_blackwell_pipeline_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=512) as (
            bbhv,
        ):
            bbh, bv = bbhv // T.ceildiv(DV, block_DV), bbhv % T.ceildiv(DV, block_DV)
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            q_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((2, block_S, block_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((2, block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")

            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_exp_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            h_tmem = T.alloc_tmem((DK, 128), dtype=accum_dtype)
            v_tmem = T.alloc_tmem((block_S, 128), dtype=accum_dtype)
            o_tmem = T.alloc_tmem((block_S, 128), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)

            mbar_u = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_v = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_p = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o0 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o1 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_h = T.alloc_barrier(arrive_count=[1] * 8)

            data_is_ready = T.alloc_barrier(arrive_count=[96] * 2)
            data_is_free = T.alloc_barrier(arrive_count=[384] * 2)
            bar_0 = T.alloc_barrier(arrive_count=384)
            bar_1 = T.alloc_barrier(arrive_count=256)
            bar_3 = T.alloc_barrier(arrive_count=128)
            bar_4 = T.alloc_barrier(arrive_count=128)
            bar_5 = T.alloc_barrier(arrive_count=384)

            T.use_swizzle(10)
            tx = T.get_thread_binding()
            num_iters = T.ceildiv(num_tokens, block_S)
            if max_iters > 0 and num_iters > max_iters:
                num_iters = max_iters

            PRODUCER_NREG = 32
            CONSUMER_V_NREG = 128
            CONSUMER_S_NREG = 160
            CONSUMER_O_NREG = 128

            if tx < 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)
                if use_initial_state:
                    T.copy(h0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], h_fragment)
                else:
                    T.clear(h_fragment)

                for i_s in T.serial(num_iters):
                    mbar_slot = i_s % 8
                    mbar_phase = (i_s // 8) % 2
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    T.barrier_wait(bar_1, i_s % 2)
                    g_last_local[0] = g_exp_shared[block_S - 1]
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] *= g_last_local[0]
                    T.copy(h_fragment, h_tmem[:, 0:block_DV])
                    T.barrier_arrive(bar_5)

                    T.barrier_wait(bar_5, i_s % 2)
                    T.tcgen05_gemm(
                        k_shared[i_s % 2, :, :],
                        vn_shared,
                        h_tmem[:, 0:block_DV],
                        transpose_A=True,
                        clear_accum=False,
                        mbar=mbar_h[mbar_slot],
                    )
                    T.mbarrier_wait_parity(mbar_h[mbar_slot], mbar_phase)
                    T.copy(h_tmem[:, 0:block_DV], h_fragment)
                    T.barrier_arrive(data_is_free[i_s % 2])

                if store_final_state:
                    T.copy(h_fragment, ht[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

            elif tx < 256:
                T.set_max_nreg(CONSUMER_V_NREG, 1)
                for i_s in T.serial(num_iters):
                    mbar_slot = i_s % 8
                    mbar_phase = (i_s // 8) % 2
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    for j_s in T.Parallel(block_S):
                        g_exp_shared[j_s] = T.exp2(
                            g_shared[i_s % 2, j_s] * 1.442695
                        )
                    for j_s in T.Parallel(block_S):
                        g_rev_exp_shared[j_s] = T.exp2(
                            (
                                g_shared[i_s % 2, block_S - 1]
                                - g_shared[i_s % 2, j_s]
                            )
                            * 1.442695
                        )
                    T.barrier_arrive(bar_1)

                    T.barrier_wait(bar_1, i_s % 2)
                    T.tcgen05_gemm(
                        k_shared[i_s % 2, :, :],
                        h_shared,
                        v_tmem[:, 0:block_DV],
                        clear_accum=True,
                        mbar=mbar_u[mbar_slot],
                    )
                    T.mbarrier_wait_parity(mbar_u[mbar_slot], mbar_phase)
                    T.copy(v_tmem[:, 0:block_DV], u_fragment)

                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                        u_fragment[j_s, j_v] += v_shared[i_s % 2, j_s, j_v]
                        v_shared[i_s % 2, j_s, j_v] = u_fragment[j_s, j_v]

                    T.barrier_wait(bar_3, i_s % 2)
                    T.tcgen05_gemm(
                        a_shared[i_s % 2, :, :],
                        v_shared[i_s % 2, :, :],
                        v_tmem[:, 0:block_DV],
                        clear_accum=True,
                        mbar=mbar_v[mbar_slot],
                    )
                    T.mbarrier_wait_parity(mbar_v[mbar_slot], mbar_phase)
                    T.copy(v_tmem[:, 0:block_DV], v_fragment)
                    T.copy(v_fragment, vd_shared)
                    T.barrier_arrive(bar_4)

                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                        vn_shared[j_s, j_v] = v_fragment[j_s, j_v]
                    T.barrier_arrive(bar_5)
                    T.barrier_wait(bar_5, i_s % 2)
                    T.barrier_arrive(data_is_free[i_s % 2])

            elif tx < 384:
                T.set_max_nreg(CONSUMER_O_NREG, 1)
                for i_s in T.serial(num_iters):
                    mbar_slot = i_s % 8
                    mbar_phase = (i_s // 8) % 2
                    left = i_s * block_S
                    right = left + block_S
                    T.barrier_wait(data_is_ready[i_s % 2], (i_s // 2) % 2)
                    T.barrier_arrive(bar_0)

                    T.barrier_wait(bar_0, i_s % 2)
                    T.tcgen05_gemm(
                        q_shared[i_s % 2, :, :],
                        k_shared[i_s % 2, :, :],
                        p_tmem,
                        transpose_B=True,
                        clear_accum=True,
                        mbar=mbar_p[mbar_slot],
                    )
                    T.mbarrier_wait_parity(mbar_p[mbar_slot], mbar_phase)
                    T.copy(p_tmem, p_fragment)

                    for j_s, j_t in T.Parallel(block_S, block_S):
                        g_fragment[j_s, j_t] = (
                            g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                        )
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        if j_s >= j_t:
                            g_fragment[j_s, j_t] = T.exp2(
                                g_fragment[j_s, j_t] * 1.442695
                            )
                        else:
                            g_fragment[j_s, j_t] = 0

                    T.barrier_wait(bar_1, i_s % 2)
                    T.tcgen05_gemm(
                        q_shared[i_s % 2, :, :],
                        h_shared,
                        o_tmem[:, 0:block_DV],
                        clear_accum=True,
                        mbar=mbar_o0[mbar_slot],
                    )
                    T.mbarrier_wait_parity(mbar_o0[mbar_slot], mbar_phase)
                    T.copy(o_tmem[:, 0:block_DV], o_fragment)

                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    T.copy(p_fragment, p_shared)
                    T.barrier_arrive(bar_3)

                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                    T.copy(o_fragment, o_tmem[:, 0:block_DV])

                    T.barrier_wait(bar_4, i_s % 2)
                    T.tcgen05_gemm(
                        p_shared,
                        vd_shared,
                        o_tmem[:, 0:block_DV],
                        clear_accum=False,
                        mbar=mbar_o1[mbar_slot],
                    )
                    T.mbarrier_wait_parity(mbar_o1[mbar_slot], mbar_phase)
                    T.copy(o_tmem[:, 0:block_DV], o_fragment)
                    T.barrier_arrive(bar_5)

                    if store_o:
                        T.copy(o_fragment, o[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV])
                    T.barrier_wait(bar_5, i_s % 2)
                    T.barrier_arrive(data_is_free[i_s % 2])

            else:
                T.set_max_nreg(PRODUCER_NREG, 0)
                if tx < 384 + 32:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S
                        T.copy(q[bb, left:right, bhg, 0:DK], q_shared[i_s % 2, :, :])
                        T.copy(k[bb, left:right, bhg, 0:DK], k_shared[i_s % 2, :, :])
                        T.barrier_arrive(data_is_ready[i_s % 2])
                elif tx < 384 + 64:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S
                        T.copy(
                            v[bb, left:right, bh, bv * block_DV : (bv + 1) * block_DV],
                            v_shared[i_s % 2, :, :],
                        )
                        T.barrier_arrive(data_is_ready[i_s % 2])
                elif tx < 384 + 96:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(data_is_free[i_s % 2], (i_s // 2 + 1) % 2)
                        left = i_s * block_S
                        right = left + block_S
                        T.copy(a[bb, left:right, bh, 0:block_S], a_shared[i_s % 2, :, :])
                        for j_s in T.Parallel(block_S):
                            g_shared[i_s % 2, j_s] = g[bb, left + j_s, bh]
                        T.barrier_arrive(data_is_ready[i_s % 2])

    return tilelang_fused_chunk_gdr_fwd_blackwell_pipeline_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_fused_chunk_gdr_fwd_blackwell_dv128_reuse_v2(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    use_initial_state,
    store_final_state,
    store_o,
    max_iters,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size
    half_DV = 64

    q_shape = (batch_size, num_tokens, Hg, DK)
    k_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    a_shape = (batch_size, num_tokens, H, chunk_size)
    g_shape = (batch_size, num_tokens, H)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)
    o_shape = (batch_size, num_tokens, H, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_blackwell_dv128_reuse_v2_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(batch_size * H, threads=256) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            q_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, half_DV), dtype=qkva_dtype)
            a_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            h_shared = T.alloc_shared((DK, half_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, half_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, half_DV), dtype=qkva_dtype)
            g_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            g_rev_exp_shared = T.alloc_shared(
                (block_S), dtype=accum_dtype, scope="shared"
            )
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

            h0_fragment = T.alloc_fragment((DK, half_DV), dtype=accum_dtype)
            h1_fragment = T.alloc_fragment((DK, half_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, half_DV), dtype=accum_dtype)
            u_fragment = T.alloc_fragment((block_S, half_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, half_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            h_tmem = T.alloc_tmem((DK, 128), dtype=accum_dtype)
            tmp_tmem = T.alloc_tmem((block_S, 128), dtype=accum_dtype)
            p_tmem = T.alloc_tmem((block_S, block_S), dtype=accum_dtype)

            mbar_u = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_v = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_p = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o0 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_o1 = T.alloc_barrier(arrive_count=[1] * 8)
            mbar_h = T.alloc_barrier(arrive_count=[1] * 8)
            bar_load = T.alloc_barrier(arrive_count=256)
            bar_h_shared = T.alloc_barrier(arrive_count=256)

            num_iters = T.ceildiv(num_tokens, block_S)
            if max_iters > 0 and num_iters > max_iters:
                num_iters = max_iters

            if use_initial_state:
                T.copy(h0[bb, bh, 0:DK, 0:half_DV], h0_fragment)
                T.copy(h0[bb, bh, 0:DK, half_DV:DV], h1_fragment)
            else:
                T.clear(h0_fragment)
                T.clear(h1_fragment)

            for i_s in T.serial(num_iters):
                left = i_s * block_S
                right = left + block_S
                mbar_slot0 = (i_s * 2) % 8
                mbar_slot1 = (i_s * 2 + 1) % 8
                mbar_phase = (i_s // 4) % 2

                T.copy(q[bb, left:right, bhg, 0:DK], q_shared)
                T.copy(k[bb, left:right, bhg, 0:DK], k_shared)
                T.copy(a[bb, left:right, bh, 0:block_S], a_shared)
                for j_s in T.Parallel(block_S):
                    g_shared[j_s] = g[bb, left + j_s, bh]
                for j_s in T.Parallel(block_S):
                    g_exp_shared[j_s] = T.exp2(g_shared[j_s] * 1.442695)
                    g_rev_exp_shared[j_s] = T.exp2(
                        (g_shared[block_S - 1] - g_shared[j_s]) * 1.442695
                    )
                T.barrier_arrive(bar_load)
                T.barrier_wait(bar_load, i_s % 2)

                for j_s, j_t in T.Parallel(block_S, block_S):
                    g_fragment[j_s, j_t] = g_shared[j_s] - g_shared[j_t]
                for j_s, j_t in T.Parallel(block_S, block_S):
                    if j_s >= j_t:
                        g_fragment[j_s, j_t] = T.exp2(
                            g_fragment[j_s, j_t] * 1.442695
                        )
                    else:
                        g_fragment[j_s, j_t] = 0

                T.tcgen05_gemm(
                    q_shared,
                    k_shared,
                    p_tmem,
                    transpose_B=True,
                    clear_accum=True,
                    mbar=mbar_p[mbar_slot0],
                )
                T.mbarrier_wait_parity(mbar_p[mbar_slot0], mbar_phase)
                T.copy(p_tmem, p_fragment)
                for j_s, j_t in T.Parallel(block_S, block_S):
                    p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                T.copy(p_fragment, p_shared)

                T.copy(v[bb, left:right, bh, 0:half_DV], v_shared)
                T.copy(h0_fragment, h_shared)
                T.barrier_arrive(bar_h_shared)
                T.barrier_wait(bar_h_shared, i_s % 2)
                T.tcgen05_gemm(
                    k_shared,
                    h_shared,
                    tmp_tmem[:, 0:half_DV],
                    clear_accum=True,
                    mbar=mbar_u[mbar_slot0],
                )
                T.mbarrier_wait_parity(mbar_u[mbar_slot0], mbar_phase)
                T.copy(tmp_tmem[:, 0:half_DV], u_fragment)
                for j_s, j_v in T.Parallel(block_S, half_DV):
                    u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                    v_shared[j_s, j_v] = u_fragment[j_s, j_v]
                T.tcgen05_gemm(
                    a_shared,
                    v_shared,
                    tmp_tmem[:, 0:half_DV],
                    clear_accum=True,
                    mbar=mbar_v[mbar_slot0],
                )
                T.mbarrier_wait_parity(mbar_v[mbar_slot0], mbar_phase)
                T.copy(tmp_tmem[:, 0:half_DV], v_fragment)
                T.copy(v_fragment, vd_shared)
                for j_s, j_v in T.Parallel(block_S, half_DV):
                    v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    vn_shared[j_s, j_v] = v_fragment[j_s, j_v]
                T.tcgen05_gemm(
                    q_shared,
                    h_shared,
                    tmp_tmem[:, 0:half_DV],
                    clear_accum=True,
                    mbar=mbar_o0[mbar_slot0],
                )
                T.mbarrier_wait_parity(mbar_o0[mbar_slot0], mbar_phase)
                T.copy(tmp_tmem[:, 0:half_DV], o_fragment)
                for j_s, j_v in T.Parallel(block_S, half_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.copy(o_fragment, tmp_tmem[:, 0:half_DV])
                T.tcgen05_gemm(
                    p_shared,
                    vd_shared,
                    tmp_tmem[:, 0:half_DV],
                    clear_accum=False,
                    mbar=mbar_o1[mbar_slot0],
                )
                T.mbarrier_wait_parity(mbar_o1[mbar_slot0], mbar_phase)
                T.copy(tmp_tmem[:, 0:half_DV], o_fragment)
                if store_o:
                    T.copy(o_fragment, o[bb, left:right, bh, 0:half_DV])
                g_last_local[0] = g_exp_shared[block_S - 1]
                for j_k, j_v in T.Parallel(DK, half_DV):
                    h0_fragment[j_k, j_v] *= g_last_local[0]
                T.copy(h0_fragment, h_tmem[:, 0:half_DV])
                T.tcgen05_gemm(
                    k_shared,
                    vn_shared,
                    h_tmem[:, 0:half_DV],
                    transpose_A=True,
                    clear_accum=False,
                    mbar=mbar_h[mbar_slot0],
                )
                T.mbarrier_wait_parity(mbar_h[mbar_slot0], mbar_phase)
                T.copy(h_tmem[:, 0:half_DV], h0_fragment)

                T.copy(v[bb, left:right, bh, half_DV:DV], v_shared)
                T.copy(h1_fragment, h_shared)
                T.barrier_arrive(bar_h_shared)
                T.barrier_wait(bar_h_shared, (i_s + 1) % 2)
                T.tcgen05_gemm(
                    k_shared,
                    h_shared,
                    tmp_tmem[:, 0:half_DV],
                    clear_accum=True,
                    mbar=mbar_u[mbar_slot1],
                )
                T.mbarrier_wait_parity(mbar_u[mbar_slot1], mbar_phase)
                T.copy(tmp_tmem[:, 0:half_DV], u_fragment)
                for j_s, j_v in T.Parallel(block_S, half_DV):
                    u_fragment[j_s, j_v] *= -g_exp_shared[j_s]
                    u_fragment[j_s, j_v] += v_shared[j_s, j_v]
                    v_shared[j_s, j_v] = u_fragment[j_s, j_v]
                T.tcgen05_gemm(
                    a_shared,
                    v_shared,
                    tmp_tmem[:, 0:half_DV],
                    clear_accum=True,
                    mbar=mbar_v[mbar_slot1],
                )
                T.mbarrier_wait_parity(mbar_v[mbar_slot1], mbar_phase)
                T.copy(tmp_tmem[:, 0:half_DV], v_fragment)
                T.copy(v_fragment, vd_shared)
                for j_s, j_v in T.Parallel(block_S, half_DV):
                    v_fragment[j_s, j_v] *= g_rev_exp_shared[j_s]
                    vn_shared[j_s, j_v] = v_fragment[j_s, j_v]
                T.tcgen05_gemm(
                    q_shared,
                    h_shared,
                    tmp_tmem[:, 0:half_DV],
                    clear_accum=True,
                    mbar=mbar_o0[mbar_slot1],
                )
                T.mbarrier_wait_parity(mbar_o0[mbar_slot1], mbar_phase)
                T.copy(tmp_tmem[:, 0:half_DV], o_fragment)
                for j_s, j_v in T.Parallel(block_S, half_DV):
                    o_fragment[j_s, j_v] *= scale * g_exp_shared[j_s]
                T.copy(o_fragment, tmp_tmem[:, 0:half_DV])
                T.tcgen05_gemm(
                    p_shared,
                    vd_shared,
                    tmp_tmem[:, 0:half_DV],
                    clear_accum=False,
                    mbar=mbar_o1[mbar_slot1],
                )
                T.mbarrier_wait_parity(mbar_o1[mbar_slot1], mbar_phase)
                T.copy(tmp_tmem[:, 0:half_DV], o_fragment)
                if store_o:
                    T.copy(o_fragment, o[bb, left:right, bh, half_DV:DV])
                for j_k, j_v in T.Parallel(DK, half_DV):
                    h1_fragment[j_k, j_v] *= g_last_local[0]
                T.copy(h1_fragment, h_tmem[:, 0:half_DV])
                T.tcgen05_gemm(
                    k_shared,
                    vn_shared,
                    h_tmem[:, 0:half_DV],
                    transpose_A=True,
                    clear_accum=False,
                    mbar=mbar_h[mbar_slot1],
                )
                T.mbarrier_wait_parity(mbar_h[mbar_slot1], mbar_phase)
                T.copy(h_tmem[:, 0:half_DV], h1_fragment)

            if store_final_state:
                T.copy(h0_fragment, ht[bb, bh, 0:DK, 0:half_DV])
                T.copy(h1_fragment, ht[bb, bh, 0:DK, half_DV:DV])

    return tilelang_fused_chunk_gdr_fwd_blackwell_dv128_reuse_v2_kernel


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
    fwd_experiment = os.environ.get("FLASHQLA_BLACKWELL_FWD_EXPERIMENT", "").lower()

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
    if (
        os.environ.get("FLASHQLA_BLACKWELL_PRETRANSFORM_A", "1") != "1"
        and fwd_experiment != "chunk_parallel"
    ):
        fallback_reasons.append("raw_a")
    use_native_by_policy, policy_reason = should_use_native_fwd(H, Hg)
    if not use_native_by_policy:
        fallback_reasons.append(policy_reason)

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
    num_threads = int(os.environ.get("FLASHQLA_BLACKWELL_FWD_THREADS", "256"))
    if num_threads not in (128, 256, 512):
        raise ValueError(
            "FLASHQLA_BLACKWELL_FWD_THREADS must be 128, 256, or 512 for the current "
            f"native fwd path, got {num_threads}"
        )
    if fwd_experiment == "chunk_parallel":
        if block_DV != 64:
            raise ValueError(
                "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=chunk_parallel currently "
                f"requires FLASHQLA_BLACKWELL_BLOCK_DV=64, got {block_DV}"
            )
        if output_h:
            raise ValueError(
                "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=chunk_parallel does not "
                "support output_h yet"
            )
        num_chunks = tilelang.cdiv(num_tokens, chunk_size)
        _debug(
            "using chunk_parallel fwd: Hopper prepare_h + Blackwell per-chunk O "
            f"H={H} Hg={Hg} chunks={num_chunks}"
        )
        h_states, prepared_final_state, _ = hopper_fused_gdr_h(
            k=k,
            v=v,
            a=a,
            g=g,
            b=b,
            initial_state=initial_state if use_initial_state else None,
            output_final_state=output_final_state,
            output_h=True,
            chunk_size=chunk_size,
            cu_seqlens=None,
        )
        tilelang_chunk_gdr_o_kernel = tilelang_chunk_gdr_o_from_h_blackwell(
            H,
            Hg,
            K,
            V,
            chunk_size,
            scale,
            qkva_dtype=q.dtype,
            g_dtype=g.dtype,
            b_dtype=b.dtype,
            h_dtype=h_states.dtype,
            o_dtype=o.dtype,
            accum_dtype="float32",
            num_threads=num_threads,
            block_DV=block_DV,
        )
        tilelang_chunk_gdr_o_kernel(q, k, v, a, g, b, h_states, o)
        final_state = prepared_final_state if output_final_state else None
        if not output_o:
            o = None
        return o, h, final_state

    if fwd_experiment == "pipeline":
        if block_DV != 64:
            raise ValueError(
                "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=pipeline currently requires "
                f"FLASHQLA_BLACKWELL_BLOCK_DV=64, got {block_DV}"
            )
        _debug("using pipeline experiment threads=512")
        tilelang_fused_chunk_gdr_fwd_kernel = (
            tilelang_fused_chunk_gdr_fwd_blackwell_pipeline(
                H,
                Hg,
                K,
                V,
                chunk_size,
                scale,
                qkva_dtype=q.dtype,
                g_dtype=g.dtype,
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
        )
        tilelang_fused_chunk_gdr_fwd_kernel(
            q,
            k,
            v,
            a,
            g,
            initial_state,
            o,
            final_state,
        )

        if not output_final_state:
            final_state = None
        if not output_o:
            o = None
        return o, h, final_state
    if fwd_experiment in ("hopper_pipeline", "hopper_port"):
        from .fused_fwd import fused_gdr_fwd as _hopper_pipeline_fwd

        _debug("using Hopper-structure Blackwell TCGEN05 pipeline")
        return _hopper_pipeline_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            g=g,
            b=b,
            scale=scale,
            initial_state=initial_state if use_initial_state else None,
            output_final_state=output_final_state,
            output_h=output_h,
            output_o=output_o,
            cu_seqlens=cu_seqlens,
            cp_seq_map=cp_seq_map,
            raw_cu_seqlens=raw_cu_seqlens,
            chunk_size=chunk_size,
        )
    if fwd_experiment == "tmem_v2":
        raise RuntimeError(
            "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=tmem_v2 is disabled: directly "
            "copying Vd from TMEM to shared corrupted the O path on B200/B300 "
            "with TileLang 0.1.9. Use the default 'ag' path while TMEM layout "
            "constraints are reworked."
        )
    if fwd_experiment == "dv128_reuse":
        raise RuntimeError(
            "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=dv128_reuse is disabled: the "
            "single-CTA two-half prototype hangs on B300 with TileLang 0.1.9, "
            "even after separate TCGEN05 mbarrier slots/parity for each half. "
            "Use the default 'ag' path while the P-reuse design is reworked as "
            "a safer two-kernel or CTA-pair schedule."
        )
    if fwd_experiment == "pg_precompute":
        raise RuntimeError(
            "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=pg_precompute is disabled: "
            "both shared-buffer and elementwise global-store variants produced "
            "intermittent O-path correctness failures on B300 with TileLang "
            "0.1.9 while final_state stayed correct. Use the default 'ag' path."
        )
    if fwd_experiment == "small_hv":
        if H <= Hg:
            raise ValueError(
                "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=small_hv is intended for "
                f"H > Hg P-reuse shapes, got H={H}, Hg={Hg}"
            )
        allow_unsafe_precompute = (
            os.environ.get("FLASHQLA_BLACKWELL_SMALL_HV_ALLOW_UNSAFE_PRECOMPUTE", "") == "1"
        )
        if (
            os.environ.get("FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P", "") != "1"
            and not allow_unsafe_precompute
        ):
            raise RuntimeError(
                "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=small_hv precompute-P/Pg "
                "paths are disabled: loading a precomputed row-major P/Pg tile "
                "back into a tcgen05 shared operand produces correctness "
                "failures on B300 with TileLang 0.1.9. Set "
                "FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P=1 to run the stable "
                "diagnostic path, or use qwen397_native for performance runs."
            )
        if block_DV != 64:
            raise ValueError(
                "FLASHQLA_BLACKWELL_FWD_EXPERIMENT=small_hv currently requires "
                f"FLASHQLA_BLACKWELL_BLOCK_DV=64, got {block_DV}"
            )
        num_chunks = tilelang.cdiv(num_tokens, chunk_size)
        recompute_p_for_debug = (
            os.environ.get("FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P", "") == "1"
        )
        _debug(
            f"using small_hv Pg-reuse path H={H} Hg={Hg} chunks={num_chunks} "
            f"recompute_p={recompute_p_for_debug}"
        )
        use_pg_input = (
            os.environ.get("FLASHQLA_BLACKWELL_SMALL_HV_USE_PG", "") == "1"
            or os.environ.get("FLASHQLA_BLACKWELL_SMALL_HV_PG_DTYPE", "").lower()
            in ("bf16", "bfloat16", "fp32", "float32")
        )
        if recompute_p_for_debug:
            p = torch.empty(
                (batch_size, num_tokens, Hg, chunk_size),
                dtype=torch.float32,
                device=q.device,
            )
            tilelang_fused_chunk_gdr_fwd_kernel = (
                tilelang_fused_chunk_gdr_fwd_blackwell_p_input(
                    H,
                    Hg,
                    K,
                    V,
                    chunk_size,
                    scale,
                    qkva_dtype=q.dtype,
                    g_dtype=g.dtype,
                    h0_dtype=initial_state.dtype,
                    ht_dtype=final_state.dtype,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_initial_state=use_initial_state,
                    store_final_state=output_final_state,
                    store_o=output_o,
                    max_iters=max_iters,
                    recompute_p_for_debug=True,
                    num_threads=num_threads,
                    block_DV=block_DV,
                )
            )
            tilelang_fused_chunk_gdr_fwd_kernel(
                q,
                k,
                v,
                a,
                g,
                p,
                initial_state,
                o,
                final_state,
            )
        elif not use_pg_input:
            p = torch.empty(
                (batch_size, num_tokens, Hg, chunk_size),
                dtype=torch.float32,
                device=q.device,
            )
            tilelang_precompute_p_kernel = tilelang_precompute_p_blackwell(
                Hg,
                K,
                chunk_size,
                accum_dtype="float32",
                qkva_dtype=q.dtype,
            )
            tilelang_precompute_p_kernel(q, k, p, num_chunks)
            tilelang_fused_chunk_gdr_fwd_kernel = (
                tilelang_fused_chunk_gdr_fwd_blackwell_p_input(
                    H,
                    Hg,
                    K,
                    V,
                    chunk_size,
                    scale,
                    qkva_dtype=q.dtype,
                    g_dtype=g.dtype,
                    h0_dtype=initial_state.dtype,
                    ht_dtype=final_state.dtype,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    use_initial_state=use_initial_state,
                    store_final_state=output_final_state,
                    store_o=output_o,
                    max_iters=max_iters,
                    recompute_p_for_debug=False,
                    num_threads=num_threads,
                    block_DV=block_DV,
                )
            )
            tilelang_fused_chunk_gdr_fwd_kernel(
                q,
                k,
                v,
                a,
                g,
                p,
                initial_state,
                o,
                final_state,
            )
        else:
            pg_dtype = (
                torch.float32
                if os.environ.get("FLASHQLA_BLACKWELL_SMALL_HV_PG_DTYPE", "").lower()
                in ("fp32", "float32")
                else q.dtype
            )
            pg = torch.empty(
                (batch_size, num_tokens, H, chunk_size),
                dtype=pg_dtype,
                device=q.device,
            )
            tilelang_precompute_pg_kernel = tilelang_precompute_pg_blackwell(
                H,
                Hg,
                K,
                chunk_size,
                scale,
                accum_dtype="float32",
                qkva_dtype=q.dtype,
                g_dtype=g.dtype,
                pg_dtype=pg.dtype,
            )
            tilelang_precompute_pg_kernel(q, k, g, pg, num_chunks)
            if os.environ.get("FLASHQLA_BLACKWELL_SMALL_HV_SYNC_PG", "") == "1":
                torch.cuda.synchronize()
            tilelang_fused_chunk_gdr_fwd_kernel = (
                tilelang_fused_chunk_gdr_fwd_blackwell_pg_input(
                    H,
                    Hg,
                    K,
                    V,
                    chunk_size,
                    scale,
                    qkva_dtype=q.dtype,
                    g_dtype=g.dtype,
                    h0_dtype=initial_state.dtype,
                    ht_dtype=final_state.dtype,
                    o_dtype=o.dtype,
                    accum_dtype="float32",
                    pg_dtype=pg.dtype,
                    use_initial_state=use_initial_state,
                    store_final_state=output_final_state,
                    store_o=output_o,
                    max_iters=max_iters,
                    num_threads=num_threads,
                    block_DV=block_DV,
                )
            )
            tilelang_fused_chunk_gdr_fwd_kernel(
                q,
                k,
                v,
                a,
                g,
                pg,
                initial_state,
                o,
                final_state,
            )
        if not output_final_state:
            final_state = None
        if not output_o:
            o = None
        return o, h, final_state
    if fwd_experiment not in ("", "ag", "small_hv", "chunk_parallel"):
        raise ValueError(
            "FLASHQLA_BLACKWELL_FWD_EXPERIMENT must be unset, 'ag', "
            "'small_hv', 'chunk_parallel', 'pg_precompute', 'dv128_reuse', "
            f"'pipeline', or 'hopper_pipeline', got {fwd_experiment!r}"
        )
    sync_barriers = _sync_barriers()
    _debug(f"threads={num_threads} sync_barriers=" + ",".join(sorted(sync_barriers)))
    tilelang_fused_chunk_gdr_fwd_kernel = tilelang_fused_chunk_gdr_fwd_blackwell_ag(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=o.dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        store_o=output_o,
        max_iters=max_iters,
        use_bar_load="load" in sync_barriers,
        use_bar_h_shared="h" in sync_barriers,
        use_bar_o="o" in sync_barriers,
        use_bar_h_scaled="hscale" in sync_barriers,
        num_threads=num_threads,
        block_DV=block_DV,
    )
    tilelang_fused_chunk_gdr_fwd_kernel(
        q,
        k,
        v,
        a,
        g,
        initial_state,
        o,
        final_state,
    )

    if not output_final_state:
        final_state = None
    if not output_o:
        o = None
    return o, h, final_state
