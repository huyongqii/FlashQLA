# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os

import torch

from flash_qla.utils import l2norm, assert_supported, is_blackwell
from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector

_cc = assert_supported()
from .hopper import (
    fused_gdr_fwd,
    fused_gdr_bwd,
    fused_gdr_h,
    kkt_solve,
)
if is_blackwell(_cc):
    from .blackwell import (  # type: ignore[no-redef]  # noqa: F401
        fused_gdr_fwd,
        fused_gdr_bwd,
        fused_gdr_h,
        kkt_solve,
    )
    from .blackwell.policy import should_use_native_fwd
from .cp_context import intra_card_cp_preprocess


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
    use_blackwell_cp = False
    if is_blackwell(_cc):
        use_blackwell_native_fwd, _ = should_use_native_fwd(H, Hg)
        blackwell_cp_requested = (
            os.environ.get("FLASHQLA_BLACKWELL_CP", "") == "1"
            or os.environ.get("FLASHQLA_BLACKWELL_CP_EXACT", "") == "1"
        )
        use_blackwell_cp = auto_cp and use_blackwell_native_fwd and blackwell_cp_requested
        if use_blackwell_cp and cu_seqlens is not None:
            raise NotImplementedError(
                "Blackwell native CP currently supports fixed-length inputs only."
            )
        if use_blackwell_cp and k.shape[0] > 1:
            use_blackwell_cp = False
        min_cp_chunks_env = os.environ.get("FLASHQLA_CP_MIN_CHUNKS", "").strip()
        if use_blackwell_cp and min_cp_chunks_env:
            min_cp_chunks = int(min_cp_chunks_env)
            if min_cp_chunks < 1:
                raise ValueError(
                    "FLASHQLA_CP_MIN_CHUNKS must be a positive integer, "
                    f"got {min_cp_chunks}"
                )
            if (k.shape[1] + 63) // 64 < min_cp_chunks:
                use_blackwell_cp = False
        if auto_cp and not blackwell_cp_requested:
            raise NotImplementedError(
                "Blackwell native intra-card CP is not implemented yet. Hopper "
                "fallback is disabled on Blackwell; rerun with --no-cp or enable "
                "FLASHQLA_BLACKWELL_CP=1."
            )
    pretransform_a = (
        os.environ.get("FLASHQLA_BLACKWELL_PRETRANSFORM_A", "1") == "1"
        and is_blackwell(_cc)
        and use_blackwell_native_fwd
        and cu_seqlens is None
        and not output_h
    )
    if use_blackwell_cp:
        use_blackwell_dual_a = (
            pretransform_a
            and os.environ.get("FLASHQLA_BLACKWELL_CP_DUAL_A", "") == "1"
        )
        if use_blackwell_dual_a:
            from .blackwell import kkt_solve_raw_and_transformed

            A_for_cp, A = kkt_solve_raw_and_transformed(k=k, b=beta, g=g)
        else:
            A_for_cp = kkt_solve(k=k, b=beta, cu_seqlens=cu_seqlens)
            A = None
        initial_state, cu_seqlens, cp_seq_map, raw_cu_seqlens = (
            intra_card_cp_preprocess(
                k=k,
                v=v,
                a=A_for_cp,
                g=g,
                b=beta,
                raw_h0=initial_state,
                raw_cu_seqlens=cu_seqlens,
            )
        )
        if A is None:
            A = kkt_solve(
                k=k,
                b=beta,
                g=g if pretransform_a else None,
                cu_seqlens=None,
            )
    else:
        kkt_kwargs = {
            "k": k,
            "b": beta,
            "cu_seqlens": cu_seqlens,
        }
        if pretransform_a:
            kkt_kwargs["g"] = g
        A = kkt_solve(**kkt_kwargs)
    if auto_cp and not use_blackwell_cp:
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
    elif not use_blackwell_cp:
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
