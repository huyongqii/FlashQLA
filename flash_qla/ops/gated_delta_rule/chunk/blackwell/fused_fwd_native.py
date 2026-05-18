# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os

import torch
import tilelang
import tilelang.language as T

from flash_qla.ops.gated_delta_rule.chunk.hopper.fused_fwd import (
    fused_gdr_fwd as hopper_fused_gdr_fwd,
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


def _tmem_width(block_DV: int) -> int:
    value = os.environ.get("FLASHQLA_BLACKWELL_TMEM_WIDTH", "128").strip().lower()
    if value in ("", "default", "128"):
        width = 128
    elif value in ("block", "block_dv", "exact"):
        width = block_DV
    else:
        width = int(value)
    if width < block_DV or width not in (block_DV, 128):
        raise ValueError(
            "FLASHQLA_BLACKWELL_TMEM_WIDTH must be 128 or one of "
            f"block/block_dv/exact for the current native fwd path, got {value!r} "
            f"with block_DV={block_DV}"
        )
    return width


def _select_block_dv(real_batch_size: int, num_v_heads: int) -> int:
    value = os.environ.get("FLASHQLA_BLACKWELL_BLOCK_DV")
    if value:
        return int(value)

    try:
        sm_count = torch.cuda.get_device_properties().multi_processor_count
    except Exception:
        sm_count = 148
    ratio = float(os.environ.get("FLASHQLA_TARGET_CTA_RATIO", "0.7"))
    target_num_ctas = max(1, int(sm_count * ratio))
    grid_size = real_batch_size * num_v_heads
    min_block_dv = int(os.environ.get("FLASHQLA_BLACKWELL_MIN_BLOCK_DV", "64"))
    if min_block_dv not in (32, 64, 128):
        raise ValueError(
            "FLASHQLA_BLACKWELL_MIN_BLOCK_DV must be 32, 64, or 128, "
            f"got {min_block_dv}"
        )
    if grid_size >= target_num_ctas:
        return 128
    if grid_size * 2 >= target_num_ctas:
        return 64
    return min_block_dv


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
    tmem_width=128,
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
            g_inv_exp_shared = T.alloc_shared(
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
            g_last_local = T.alloc_local((1), dtype=accum_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

            # Keep TMEM allocations reusable. Blackwell TCGEN05 accepts wider
            # output tiles than this kernel stores; expose the width so exact
            # TMEM can be tested without changing the default benchmark path.
            h_tmem = T.alloc_tmem((DK, tmem_width), dtype=accum_dtype)
            tmp_tmem = T.alloc_tmem((block_S, tmem_width), dtype=accum_dtype)
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

            T.use_swizzle(10)

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
                    g_inv_exp_shared[j_s] = 1.0 / g_exp_shared[j_s]
                if use_bar_load:
                    T.barrier_arrive(bar_load)
                    T.barrier_wait(bar_load, i_s % 2)
                for j_s in T.Parallel(block_S):
                    g_rev_exp_shared[j_s] = (
                        g_exp_shared[block_S - 1] * g_inv_exp_shared[j_s]
                    )

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
                    if j_s >= j_t:
                        p_fragment[j_s, j_t] *= (
                            scale * g_exp_shared[j_s] * g_inv_exp_shared[j_t]
                        )
                    else:
                        p_fragment[j_s, j_t] = 0
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
    if cp_seq_map is not None:
        fallback_reasons.append("cp_seq_map")
    if cu_seqlens is not None:
        fallback_reasons.append("varlen")
    if num_tokens % chunk_size != 0:
        fallback_reasons.append("ragged_tokens")
    if os.environ.get("FLASHQLA_BLACKWELL_PRETRANSFORM_A", "1") != "1":
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

    block_DV = _select_block_dv(real_batch_size, H)
    if block_DV not in (32, 64, 128):
        raise ValueError(
            "FLASHQLA_BLACKWELL_BLOCK_DV must be 32, 64, or 128 for the current "
            f"TileLang 0.1.9 TCGEN05 path, got {block_DV}"
        )
    tmem_width = _tmem_width(block_DV)
    max_iters = int(os.environ.get("FLASHQLA_BLACKWELL_FWD_MAX_ITERS", "0"))
    if max_iters > 0:
        _debug(f"debug max_iters={max_iters}; output is partial and benchmark is invalid")
    num_threads = int(os.environ.get("FLASHQLA_BLACKWELL_FWD_THREADS", "256"))
    if num_threads not in (128, 256, 512):
        raise ValueError(
            "FLASHQLA_BLACKWELL_FWD_THREADS must be 128, 256, or 512 for the current "
            f"native fwd path, got {num_threads}"
        )
    if fwd_experiment not in ("", "ag"):
        raise ValueError(
            "FLASHQLA_BLACKWELL_FWD_EXPERIMENT must be unset or 'ag' for "
            f"the cleaned Blackwell native path, got {fwd_experiment!r}"
        )
    sync_barriers = _sync_barriers()
    _debug(
        f"threads={num_threads} block_DV={block_DV} tmem_width={tmem_width} "
        "sync_barriers=" + ",".join(sorted(sync_barriers))
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
        tmem_width=tmem_width,
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
