# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import math
import os

import torch
import tilelang

from flash_qla.utils import tensor_cache, assert_supported, is_blackwell

_cc = assert_supported()
if is_blackwell(_cc):
    from .blackwell.prepare_h import fused_gdr_h
    from .blackwell.cp_fwd import correct_initial_states, get_warmup_chunks
else:
    from .hopper import get_warmup_chunks, fused_gdr_h, correct_initial_states


MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count


@tensor_cache
def _create_cu_seqlens(
    batch_size: int,
    num_tokens: int,
    device_idx: int,
):
    return (
        torch.arange((batch_size + 1), dtype=torch.int32, device=f"cuda:{device_idx}")
        * num_tokens
    )


@tensor_cache
def _calc_cp_seqs(
    raw_cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_v_heads: int,
):
    # TODO: tilelang kernel
    device = raw_cu_seqlens.device
    seqlen_dtype = raw_cu_seqlens.dtype
    raw_cu_seqlens = raw_cu_seqlens.tolist()
    raw_batch_size = len(raw_cu_seqlens) - 1
    seqlens = [raw_cu_seqlens[i + 1] - raw_cu_seqlens[i] for i in range(raw_batch_size)]
    num_chunks = [tilelang.cdiv(x, chunk_size) for x in seqlens]

    # autocp
    H = num_v_heads
    # Latency model: T = a·L_cp + b·(B·H·Lc/P) / L_cp + c
    # Minimizing T yields the theoretical optimum: L_cp* ∝ √(B·H·Lc / P), where P = MULTI_PROCESSOR_COUNT, L_cp = max_local_chunks
    # Scaled by empirical factor (3) and aligned to the nearest power of 2 for optimal SM scheduling & memory alignment.

    max_local_chunks_env = os.environ.get("FLASHQLA_CP_MAX_LOCAL_CHUNKS", "").strip()
    if max_local_chunks_env:
        max_local_chunks = int(max_local_chunks_env)
        if max_local_chunks < 1:
            raise ValueError(
                "FLASHQLA_CP_MAX_LOCAL_CHUNKS must be a positive integer, "
                f"got {max_local_chunks}"
            )
    else:
        max_local_chunks = 2 ** round(
            math.log2(math.sqrt(H * sum(num_chunks) / MULTI_PROCESSOR_COUNT) * 3)
        )
        # Set min to 4 to ensure multi-stage pipelining in fused_gdr.
        max_local_chunks = max(max_local_chunks, 4)

    use_cp = False
    cp_cu_seqlens = []
    ht_mask = []
    seq_map_c2r = []
    seq_map_r2c = [0]
    max_local_tokens = max_local_chunks * chunk_size
    for i, c in enumerate(num_chunks):
        s = raw_cu_seqlens[i]
        e = raw_cu_seqlens[i + 1]
        if c > max_local_chunks:
            while s < e:
                cp_cu_seqlens.append(s)
                ht_mask.append(False)
                seq_map_c2r.append(i)
                s += max_local_tokens
            ht_mask[-1] = True
        else:
            cp_cu_seqlens.append(s)
            ht_mask.append(True)
            seq_map_c2r.append(i)
        seq_map_r2c.append(len(cp_cu_seqlens))
    cp_cu_seqlens.append(raw_cu_seqlens[-1])

    # Disable CP when B * H naturally saturates SM occupancy.
    # For varlen inputs, use `total_chunks / max_seq_chunks` as effective B,
    # since CP helps accelerate highly uneven sequence lengths.

    Be = sum(num_chunks) / max(num_chunks)
    use_cp = Be * H <= 40 or (Be * H <= 56 and max(num_chunks) >= 128)
    min_chunks_env = os.environ.get("FLASHQLA_CP_MIN_CHUNKS", "").strip()
    if min_chunks_env:
        min_chunks = int(min_chunks_env)
        if min_chunks < 1:
            raise ValueError(
                "FLASHQLA_CP_MIN_CHUNKS must be a positive integer, "
                f"got {min_chunks}"
            )
        use_cp = use_cp and max(num_chunks) >= min_chunks

    # Allow forcibly disabling/enabling CP for benchmarking on new archs.
    _cp_env = os.environ.get("FLASHQLA_AUTOCP", "").strip()
    if _cp_env == "0":
        use_cp = False
    elif _cp_env == "1":
        use_cp = True

    if use_cp:
        cp_cu_seqlens = torch.tensor(
            cp_cu_seqlens, dtype=seqlen_dtype, device=device, requires_grad=False
        )
        seq_map_c2r = torch.tensor(seq_map_c2r, dtype=seqlen_dtype, device=device)
        seq_map_r2c = torch.tensor(
            seq_map_r2c, dtype=seqlen_dtype, device=device, requires_grad=False
        )
        ht_mask = torch.tensor(
            ht_mask, dtype=torch.bool, device=device, requires_grad=False
        )
    else:
        cp_cu_seqlens, seq_map_r2c, ht_mask = None, None, None

    return use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask


def intra_card_cp_preprocess(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    raw_h0: torch.Tensor,
    raw_cu_seqlens: torch.Tensor,
    warmup_threshold: float = -10.0,
):
    batch_size, num_tokens, num_k_heads, k_head_dim = k.shape
    _, _, num_v_heads, v_head_dim = v.shape
    chunk_size = a.shape[-1]
    device = k.device

    if batch_size > 1:
        return raw_h0, raw_cu_seqlens, None, None

    if raw_cu_seqlens is None:
        device_idx = device.index
        if device_idx is None:
            device_idx = torch.cuda.current_device()
        raw_cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, device_idx)

    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r, ht_mask = _calc_cp_seqs(
        raw_cu_seqlens,
        chunk_size,
        num_v_heads,
    )

    if not use_cp:
        return raw_h0, raw_cu_seqlens, None, None

    threshold_env = os.environ.get("FLASHQLA_CP_WARMUP_THRESHOLD", "").strip()
    if threshold_env:
        warmup_threshold = float(threshold_env)
    if os.environ.get("FLASHQLA_CP_EXACT", "") == "1":
        warmup_threshold = float("-inf")

    num_warmup_chunks, fallback_mask = get_warmup_chunks(
        g=g,
        cu_seqlens=cp_cu_seqlens,
        ht_mask=ht_mask,
        chunk_size=chunk_size,
        threshold=warmup_threshold,
    )  # [cp_batch_size, num_v_heads]
    _, ht, mt = fused_gdr_h(
        k=k,
        v=v,
        a=a,
        g=g,
        b=b,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cp_cu_seqlens,
        num_warmup_chunks=num_warmup_chunks,
    )  # [cp_batch_size, num_v_heads, k_head_dim, v_head_dim]
    if os.environ.get("FLASHQLA_CP_CORRECT_H0_TORCH", "") == "1":
        cp_h0 = _correct_initial_states_torch(
            raw_h0=raw_h0,
            ht_buffer=ht,
            mt_buffer=mt,
            fallback_mask=fallback_mask,
            seq_map_r2c=seq_map_r2c,
        )
    else:
        cp_h0 = correct_initial_states(
            raw_h0=raw_h0,
            ht_buffer=ht,
            mt_buffer=mt,
            fallback_mask=fallback_mask,
            seq_map_r2c=seq_map_r2c,
        )

    return cp_h0, cp_cu_seqlens, seq_map_c2r, raw_cu_seqlens


def _correct_initial_states_torch(
    raw_h0: torch.Tensor | None,
    ht_buffer: torch.Tensor,
    mt_buffer: torch.Tensor,
    fallback_mask: torch.Tensor,
    seq_map_r2c: torch.Tensor,
) -> torch.Tensor:
    """Debug-only PyTorch reference for CP h0 correction.

    This intentionally mirrors ``tilelang_correct_h0`` and is gated by
    FLASHQLA_CP_CORRECT_H0_TORCH=1. It is useful on new architectures when we
    need to separate CP math issues from TileLang codegen/synchronization bugs.
    """

    cp_batch_size, num_heads, k_head_dim, v_head_dim = ht_buffer.shape
    raw_batch_size = seq_map_r2c.numel() - 1
    res_dtype = torch.float32 if raw_h0 is None else raw_h0.dtype
    cp_h0 = torch.empty(
        (cp_batch_size, num_heads, k_head_dim, v_head_dim),
        dtype=res_dtype,
        device=ht_buffer.device,
    )

    for raw_b in range(raw_batch_size):
        start = int(seq_map_r2c[raw_b].item())
        end = int(seq_map_r2c[raw_b + 1].item())
        if raw_h0 is None:
            state = torch.zeros(
                (num_heads, k_head_dim, v_head_dim),
                dtype=torch.float32,
                device=ht_buffer.device,
            )
        else:
            state = raw_h0[raw_b].to(torch.float32)
        cp_h0[start] = state.to(res_dtype)

        for cp_b in range(start, end - 1):
            transformed = torch.matmul(
                mt_buffer[cp_b].to(torch.float32),
                state,
            )
            next_state = ht_buffer[cp_b].to(torch.float32)
            use_transform = fallback_mask[cp_b].view(num_heads, 1, 1)
            state = torch.where(use_transform, next_state + transformed, next_state)
            cp_h0[cp_b + 1] = state.to(res_dtype)

    return cp_h0
