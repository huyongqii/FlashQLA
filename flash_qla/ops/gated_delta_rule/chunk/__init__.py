# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os

import torch
import tilelang

from flash_qla.utils import l2norm, assert_supported, is_blackwell
from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector

_cc = assert_supported()
from .hopper import (
    correct_initial_states,
    fused_gdr_fwd,
    fused_gdr_bwd,
    fused_gdr_h,
    kkt_solve,
)
if is_blackwell(_cc):
    from .blackwell import (  # type: ignore[no-redef]  # noqa: F401
        correct_initial_states,
        fused_gdr_fwd,
        fused_gdr_bwd,
        fused_gdr_h,
        kkt_solve,
    )
    from .blackwell.policy import should_use_native_fwd
from .cp_context import intra_card_cp_preprocess


def _blackwell_segment_chunks() -> int:
    value = os.environ.get("FLASHQLA_BLACKWELL_SEGMENT_CHUNKS", "").strip()
    if not value:
        return 0
    segment_chunks = int(value)
    if segment_chunks < 1:
        raise ValueError(
            "FLASHQLA_BLACKWELL_SEGMENT_CHUNKS must be a positive integer, "
            f"got {segment_chunks}"
        )
    return segment_chunks


def _debug_blackwell_segment(message: str):
    if os.environ.get("FLASHQLA_DEBUG_BLACKWELL_DISPATCH", "") == "1":
        print(f"[FlashQLA Blackwell segment fwd] {message}", flush=True)


def _try_blackwell_segmented_fwd(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    output_h: bool,
    cu_seqlens: torch.LongTensor | None,
    auto_cp: bool,
    chunk_size: int,
):
    segment_chunks = _blackwell_segment_chunks()
    if segment_chunks == 0:
        return None
    if output_h or cu_seqlens is not None or auto_cp:
        _debug_blackwell_segment("disabled reason=output_h_or_varlen_or_auto_cp")
        return None
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    if batch_size != 1 or K != 128 or V != 128 or chunk_size != 64:
        _debug_blackwell_segment("disabled reason=unsupported_shape")
        return None
    segment_tokens = segment_chunks * chunk_size
    if num_tokens % segment_tokens != 0:
        _debug_blackwell_segment("disabled reason=ragged_segments")
        return None

    num_segments = num_tokens // segment_tokens
    if num_segments <= 1:
        _debug_blackwell_segment("disabled reason=single_segment")
        return None

    seqlen_dtype = torch.int32
    cu_segments = (
        torch.arange(num_segments + 1, dtype=seqlen_dtype, device=k.device)
        * segment_tokens
    )
    num_warmup_chunks = torch.full(
        (num_segments, H),
        segment_chunks,
        dtype=seqlen_dtype,
        device=k.device,
    )
    fallback_mask = torch.ones(
        (num_segments, H),
        dtype=torch.bool,
        device=k.device,
    )

    _debug_blackwell_segment(
        f"using segmented native fwd segments={num_segments} "
        f"segment_chunks={segment_chunks}"
    )
    _, ht, mt = fused_gdr_h(
        k=k,
        v=v,
        a=a,
        g=g,
        b=b,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cu_segments,
        num_warmup_chunks=num_warmup_chunks,
    )
    seq_map_r2c = torch.tensor([0, num_segments], dtype=seqlen_dtype, device=k.device)
    segment_initial_state = correct_initial_states(
        raw_h0=initial_state,
        ht_buffer=ht,
        mt_buffer=mt,
        fallback_mask=fallback_mask,
        seq_map_r2c=seq_map_r2c,
    )

    q_segmented = q.reshape(num_segments, segment_tokens, Hg, K)
    k_segmented = k.reshape(num_segments, segment_tokens, Hg, K)
    v_segmented = v.reshape(num_segments, segment_tokens, H, V)
    a_segmented = a.reshape(num_segments, segment_tokens, H, chunk_size)
    g_segmented = g.reshape(num_segments, segment_tokens, H)
    b_segmented = b.reshape(num_segments, segment_tokens, H)
    o_segmented, h_segmented, final_state_segmented = fused_gdr_fwd(
        q=q_segmented,
        k=k_segmented,
        v=v_segmented,
        a=a_segmented,
        g=g_segmented,
        b=b_segmented,
        scale=scale,
        initial_state=segment_initial_state,
        output_final_state=output_final_state,
        output_h=False,
        output_o=True,
        cu_seqlens=None,
        cp_seq_map=None,
        raw_cu_seqlens=None,
        chunk_size=chunk_size,
    )
    o = o_segmented.reshape(batch_size, num_tokens, H, V)
    h = torch.empty((batch_size, 0, H, K, V), dtype=k.dtype, device=k.device)
    final_state = None
    if output_final_state:
        final_state = final_state_segmented[-batch_size:].contiguous()
    return o, h, final_state


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    output_final_state: bool = True,
    output_h: bool = False,
    auto_cp: bool = True,
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    Hg, H = k.shape[-2], v.shape[-2]
    use_blackwell_native_fwd = False
    if is_blackwell(_cc):
        use_blackwell_native_fwd, _ = should_use_native_fwd(H, Hg)
    pretransform_a = (
        os.environ.get("FLASHQLA_BLACKWELL_PRETRANSFORM_A", "1") == "1"
        and is_blackwell(_cc)
        and use_blackwell_native_fwd
        and cu_seqlens is None
        and not output_h
        and not auto_cp
    )
    kkt_kwargs = {
        "k": k,
        "b": beta,
        "cu_seqlens": cu_seqlens,
    }
    if pretransform_a:
        kkt_kwargs["g"] = g
    A = kkt_solve(**kkt_kwargs)
    if os.environ.get("FLASHQLA_BLACKWELL_PRECOMPUTE_P", "") == "1":
        raise RuntimeError(
            "FLASHQLA_BLACKWELL_PRECOMPUTE_P=1 is disabled: the first prototype "
            "corrupted final_state on B200. Use the default pretransform-A path "
            "while the P-reuse design is reworked."
        )
    segmented_result = None
    if is_blackwell(_cc) and use_blackwell_native_fwd:
        segmented_result = _try_blackwell_segmented_fwd(
            q=q,
            k=k,
            v=v,
            a=A,
            g=g,
            b=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_h=output_h,
            cu_seqlens=cu_seqlens,
            auto_cp=auto_cp,
            chunk_size=64,
        )
    if segmented_result is not None:
        o, h, final_state = segmented_result
        return g, A, o, h, final_state
    if auto_cp:
        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
            intra_card_cp_preprocess(
                k=k,
                v=v,
                a=A,
                g=g,
                b=beta,
                raw_h0=initial_state,
                raw_cu_seqlens=cu_seqlens,
            )
        )
    else:
        cp_seq_map = None
        raw_cu_seqlens = None
    fwd_kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "a": A,
        "g": g,
        "b": beta,
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        "output_h": output_h,
        "output_o": True,
        "cu_seqlens": cu_seqlens,
        "cp_seq_map": cp_seq_map,
        "raw_cu_seqlens": raw_cu_seqlens,
    }
    o, h, final_state = fused_gdr_fwd(**fwd_kwargs)
    return g, A, o, h, final_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    h, _, _ = fused_gdr_h(
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        initial_state=initial_state,
        output_final_state=False,
        output_h=True,
        cu_seqlens=cu_seqlens,
    )
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        do=do,
        dht=dht,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        dq = group_reduce_vector(dq, Hg)
        dk = group_reduce_vector(dk, Hg)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        q_orig = q
        k_orig = k

        g, A, o, _, final_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            output_h=False,
            cu_seqlens=cu_seqlens,
        )

        ctx.save_for_backward(q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        return o.to(q.dtype), final_state

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q_orig,
            k=k_orig,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq.to(q_orig),
            dk.to(k_orig),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
        )


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
):
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16 or float16."
    )
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0, (
        "num_qk_heads must be divisible to num_v_heads."
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    )

    return o, final_state
