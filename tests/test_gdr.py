# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import argparse
import fnmatch
import math
import os

import torch
import pandas as pd

# Requires flash-linear-attention==0.5.0
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_fla,
)
from fla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_fla,
)

from flash_qla import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_qla
from flash_qla import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_qla
from flash_qla.utils import l2norm, pack, profile

from ref_gdr import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_ref
from ref_gdr import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_ref


def _blackwell_native_fwd_requested() -> bool:
    kernels = {
        item.strip().lower()
        for item in os.environ.get("FLASHQLA_BLACKWELL_NATIVE_KERNELS", "").split(",")
        if item.strip()
    }
    return (
        os.environ.get("FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE", "") == "1"
        and os.environ.get("FLASHQLA_BLACKWELL_NATIVE", "") == "1"
        and ("fwd" in kernels or "all" in kernels)
    )


def _profile_value(prof: dict[str, float], label: str, *kernel_groups):
    value = 0.0
    missing = []
    for kernel_group in kernel_groups:
        if isinstance(kernel_group, str):
            kernel_group = (kernel_group,)
        matches = []
        for kernel_name in kernel_group:
            if "*" in kernel_name:
                matches.extend(
                    value
                    for key, value in prof.items()
                    if fnmatch.fnmatch(key, kernel_name)
                )
            elif kernel_name in prof:
                matches.append(prof[kernel_name])
        if matches:
            value += sum(matches)
        else:
            missing.append(kernel_group[0])
    if missing:
        print(f"{label}: missing profiler kernel(s): {missing}")
        return None
    return value


def _has_kernel_events(prof: dict[str, float]) -> bool:
    ignored_prefixes = ("aten::", "cuda", "cu", "ProfilerStep")
    return any(
        key != "total" and not key.startswith(ignored_prefixes)
        for key in prof.keys()
    )


def _print_profiler_note(label: str, prof: dict[str, float]):
    if _has_kernel_events(prof):
        return
    print(
        f"{label}: torch.profiler did not report GPU kernel events; "
        "per-kernel rows are shown as NaN, but total latency still comes "
        "from tilelang.profiler.do_bench(). This is commonly caused by CUPTI "
        "subscriber conflicts or library/profiler incompatibility."
    )


def _skip_fla_bwd_by_default(device: torch.device | str) -> bool:
    force = os.environ.get("FLASHQLA_SKIP_FLA_BWD", "").strip()
    if force:
        return force == "1"
    if not torch.cuda.is_available():
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False
    device_index = torch.cuda.current_device() if device.index is None else device.index
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 10


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _print_error_location(label: str, actual: torch.Tensor, expected: torch.Tensor, chunk_size: int):
    diff = (actual.float() - expected.float()).abs()
    flat_idx = int(diff.argmax().item())
    idx = tuple(
        int(i)
        for i in torch.unravel_index(
            torch.tensor(flat_idx, device=diff.device), diff.shape
        )
    )
    max_abs = diff[idx].item()
    denom = expected.float().abs().amax().clamp_min(1e-6)
    print(
        f"{label} max_idx={idx} max_abs={max_abs:.6f} "
        f"rel={(diff[idx] / denom).item():.6f} "
        f"actual={actual[idx].float().item():.6f} "
        f"expected={expected[idx].float().item():.6f}"
    )
    if diff.ndim == 4:
        # O layout: [B, T, H, D]
        per_head = diff.amax(dim=(0, 1, 3))
        top_heads = torch.topk(per_head, k=min(8, per_head.numel()))
        print(
            f"{label} top_heads="
            + ", ".join(
                f"h{int(h)}:{float(v):.6f}"
                for v, h in zip(top_heads.values, top_heads.indices)
            )
        )
        num_chunks = diff.shape[1] // chunk_size
        if num_chunks > 0:
            chunk_diff = diff[:, : num_chunks * chunk_size].reshape(
                diff.shape[0], num_chunks, chunk_size, diff.shape[2], diff.shape[3]
            )
            per_chunk = chunk_diff.amax(dim=(0, 2, 3, 4))
            top_chunks = torch.topk(per_chunk, k=min(8, per_chunk.numel()))
            print(
                f"{label} top_chunks="
                + ", ".join(
                    f"c{int(c)}:{float(v):.6f}"
                    for v, c in zip(top_chunks.values, top_chunks.indices)
                )
            )


def test_gated_delta_rule(
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    varlen: bool = False,
    cu_seqlens: list[int] | None = None,
    use_h0: bool = False,
    chunk_size: int = 64,
    data_dtype: str = "bfloat16",
    ref_dtype: str = "float32",
    device: torch.device = "cuda",
    random_seed: int = 42,
    check_accuracy: bool = True,
    show_speedup: bool = True,
    auto_cp: bool = True,
    swa_ratio: float = 0.75,
    skip_bwd: bool = False,
    skip_fla_bwd: bool | None = None,
):
    data_dtype = getattr(torch, data_dtype)
    ref_dtype = getattr(torch, ref_dtype)
    correctness_repeats = max(1, _env_int("FLASHQLA_CORRECTNESS_REPEATS", 1000))
    torch.manual_seed(random_seed)
    q = l2norm(
        torch.randn(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            device=device,
            dtype=data_dtype,
        )
    )
    k = l2norm(
        torch.randn(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            device=device,
            dtype=data_dtype,
        )
    )
    v = torch.randn(
        (batch_size, num_tokens, num_v_heads, head_dim_v),
        device=device,
        dtype=data_dtype,
    )
    g = (
        torch.nn.functional.logsigmoid(
            torch.randn(
                (batch_size, num_tokens, num_v_heads),
                device=device,
                dtype=torch.float32,
            )
        )
        / 16
    )
    beta = torch.randn(
        (batch_size, num_tokens, num_v_heads), device=device, dtype=torch.float32
    ).sigmoid()
    h0 = (
        torch.randn(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            device=device,
            dtype=torch.float32,
        )
        if use_h0
        else None
    )
    do = torch.randn_like(v)
    dht = (
        torch.randn(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            device=device,
            dtype=torch.float32,
        )
        / 8
        if use_h0
        else None
    )
    scale = head_dim_k ** (-0.5)
    print(
        f"Shape: B={batch_size} Hk={num_k_heads} Hv={num_v_heads} T={num_tokens} VarLen={varlen}"
    )

    swa_mask = torch.zeros((num_v_heads), dtype=torch.bool, device=device)
    swa_mask[: math.ceil(swa_ratio * num_v_heads)] = 1
    swa_mask = swa_mask[torch.randperm(num_v_heads, device=device)]
    g[:, :, ~swa_mask] = 0.0
    print(f"SWA Mask: {swa_mask.to(torch.int32, copy=True).tolist()}")

    if varlen:
        if cu_seqlens is None:
            cu_seqlens = torch.randint(
                1, num_tokens, (batch_size,), device=device, dtype=torch.int32
            )
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(cu_seqlens, dim=-1), (1, 0)
            )
            q = pack(q, cu_seqlens)
            k = pack(k, cu_seqlens)
            v = pack(v, cu_seqlens)
            g = pack(g, cu_seqlens)
            beta = pack(beta, cu_seqlens)
            do = pack(do, cu_seqlens)
        else:
            assert batch_size == 1
            assert cu_seqlens[0] == 0
            assert cu_seqlens[-1] == num_tokens
            cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
            if use_h0:
                real_batch_size = cu_seqlens.shape[0] - 1
                h0 = torch.randn(
                    (real_batch_size, num_v_heads, head_dim_k, head_dim_v),
                    device=device,
                    dtype=torch.float32,
                )
                dht = (
                    torch.randn(
                        (real_batch_size, num_v_heads, head_dim_k, head_dim_v),
                        device=device,
                        dtype=torch.float32,
                    )
                    / 8
                )
            assert (cu_seqlens[1:] - cu_seqlens[:-1]).min() > 0
    else:
        cu_seqlens = None

    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.to(ref_dtype, copy=True),
        k=k.to(ref_dtype, copy=True),
        v=v.to(ref_dtype, copy=True),
        g=g.to(ref_dtype, copy=True),
        beta=beta.to(ref_dtype, copy=True),
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
    )
    g_fla, o_fla, A_fla, s_fla, _, _ = chunk_gated_delta_rule_fwd_fla(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    qla_output_h = not _blackwell_native_fwd_requested()
    g_qla, A_qla, o_qla, h_qla, s_qla = chunk_gated_delta_rule_fwd_qla(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        output_h=qla_output_h,
        auto_cp=auto_cp,
    )

    if check_accuracy:
        if h_qla is not None and h_qla.numel() > 0 and h_qla.shape == h_ref.shape:
            print(
                f"h_qla: {(h_qla - h_ref).abs().max().item():.4f} / {h_ref.abs().max().item():.4f}"
            )
        else:
            print("h_qla: skipped (Blackwell native fwd output_h unsupported)")
        print(
            f"s_fla: {(s_fla - s_ref).abs().max().item():.4f} / {s_ref.abs().max().item():.4f}"
        )
        print(
            f"s_qla: {(s_qla - s_ref).abs().max().item():.4f} / {s_ref.abs().max().item():.4f}"
        )
        print(
            f"o_fla: {(o_fla - o_ref).abs().max().item():.4f} / {o_ref.abs().max().item():.4f}"
        )
        print(
            f"o_qla: {(o_qla - o_ref).abs().max().item():.4f} / {o_ref.abs().max().item():.4f}"
        )

        for repeat_idx in range(correctness_repeats):
            g_qla, A_qla, o_qla, h_qla, s_qla = chunk_gated_delta_rule_fwd_qla(
                q,
                k,
                v,
                g,
                beta,
                scale,
                h0,
                cu_seqlens,
                True,
                False,
                auto_cp,
            )
            try:
                if h0 is not None:
                    assert (
                        s_qla - s_ref
                    ).abs().max().item() <= s_ref.abs().max().item() * 0.02
                assert (
                    o_qla - o_ref
                ).abs().max().item() <= o_ref.abs().max().item() * 0.02
            except AssertionError as e:
                print("********** ERROR **********")
                print(f"fwd correctness repeat: {repeat_idx + 1}/{correctness_repeats}")
                if h0 is not None:
                    print(
                        f"s_qla: {(s_qla - s_ref).abs().max().item():.4f} / {s_ref.abs().max().item():.4f}"
                    )
                print(
                    f"o_qla: {(o_qla - o_ref).abs().max().item():.4f} / {o_ref.abs().max().item():.4f}"
                )
                _print_error_location("o_qla", o_qla, o_ref, chunk_size)
                if h0 is not None:
                    _print_error_location("s_qla", s_qla, s_ref, chunk_size)
                print("********** ERROR **********")
                raise e

    if show_speedup:
        prof_fla = profile(
            chunk_gated_delta_rule_fwd_fla,
            [q, k, v, g, beta, scale, h0, True, cu_seqlens],
        )
        prof_qla = profile(
            chunk_gated_delta_rule_fwd_qla,
            [q, k, v, g, beta, scale, h0, cu_seqlens, True, False, auto_cp],
        )
        print(f"[fwd] prof_fla keys ({len(prof_fla)}): {sorted(prof_fla.keys())}")
        print(f"[fwd] prof_qla keys ({len(prof_qla)}): {sorted(prof_qla.keys())}")
        _print_profiler_note("[fwd] FLA", prof_fla)
        _print_profiler_note("[fwd] FlashQLA", prof_qla)
        result_fla = {
            "[fwd] csum": _profile_value(
                prof_fla, "[fwd] FLA csum", "chunk_local_cumsum_scalar_kernel"
            ),
            "[fwd] solve": _profile_value(
                prof_fla,
                "[fwd] FLA solve",
                "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
            ),
            "[fwd] wu": _profile_value(
                prof_fla, "[fwd] FLA wu", "recompute_w_u_fwd_kernel"
            ),
            "[fwd] gdr": _profile_value(
                prof_fla,
                "[fwd] FLA gdr",
                "chunk_gated_delta_rule_fwd_kernel_h_blockdim*",
            ),
            "[fwd] o": _profile_value(prof_fla, "[fwd] FLA o", "chunk_fwd_kernel_o"),
        }
        result_qla = {
            "[fwd] csum": _profile_value(
                prof_qla,
                "[fwd] FlashQLA csum",
                "tilelang_chunk_local_cumsum_kernel_kernel",
            ),
            "[fwd] solve": _profile_value(
                prof_qla,
                "[fwd] FlashQLA solve",
                (
                    "tilelang_kkt_solve_kernel_kernel",
                    "tilelang_kkt_solve_fixed_fast_kernel_kernel",
                ),
            ),
            "[fwd] gdr": _profile_value(
                prof_qla,
                "[fwd] FlashQLA gdr",
                (
                    "tilelang_fused_chunk_gdr_fwd_kernel_kernel",
                    "tilelang_fused_chunk_gdr_fwd_blackwell_native_kernel_kernel",
                    "tilelang_fused_chunk_gdr_fwd_blackwell_ag_kernel_kernel",
                ),
            ),
            "[fwd] o": 0.0,
        }
        if (
            "tilelang_get_warmup_chunks_kernel_kernel" in prof_qla.keys()
            or "tilelang_prepare_h_kernel_kernel" in prof_qla.keys()
            or "tilelang_transform_a_kernel_kernel" in prof_qla.keys()
        ):
            result_fla["[fwd] cp-w"] = None
            result_fla["[fwd] cp-a"] = None
            result_fla["[fwd] cp-h"] = None
            result_fla["[fwd] cp-c"] = None
            result_qla["[fwd] cp-w"] = _profile_value(
                prof_qla,
                "[fwd] FlashQLA cp-w",
                "tilelang_get_warmup_chunks_kernel_kernel",
            )
            result_qla["[fwd] cp-a"] = _profile_value(
                prof_qla, "[fwd] FlashQLA cp-a", "tilelang_transform_a_kernel_kernel"
            )
            result_qla["[fwd] cp-h"] = _profile_value(
                prof_qla, "[fwd] FlashQLA cp-h", "tilelang_prepare_h_kernel_kernel"
            )
            result_qla["[fwd] cp-c"] = _profile_value(
                prof_qla, "[fwd] FlashQLA cp-c", "tilelang_correct_h0_kernel_kernel"
            )
        result_fla["total"] = prof_fla["total"]
        result_qla["total"] = prof_qla["total"]
        results = {
            "fla": result_fla,
            "flash_qla": result_qla,
        }
        df = pd.DataFrame(results)
        print(df.round(3))
        speedup = results["fla"]["total"] / results["flash_qla"]["total"]
        print(f"Speed up: {speedup:.2f}x")

    if skip_bwd:
        return
    if skip_fla_bwd is None:
        skip_fla_bwd = _skip_fla_bwd_by_default(device)

    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(ref_dtype, copy=True),
        k.to(ref_dtype, copy=True),
        v.to(ref_dtype, copy=True),
        g_ref,
        beta.to(ref_dtype, copy=True),
        A_ref.to(ref_dtype, copy=True),
        scale,
        h0,
        do.to(ref_dtype, copy=True),
        dht,
        cu_seqlens,
    )
    if skip_fla_bwd:
        print(
            "[bwd] skip FLA backward baseline "
            "(set FLASHQLA_SKIP_FLA_BWD=0 to force it)."
        )
        dq_fla = dk_fla = dv_fla = db_fla = dg_fla = dh0_fla = None
    else:
        dq_fla, dk_fla, dv_fla, db_fla, dg_fla, dh0_fla, _, _ = (
            chunk_gated_delta_rule_bwd_fla(
                q,
                k,
                v,
                g_fla,
                beta,
                A_fla,
                scale,
                h0,
                do,
                dht,
                cu_seqlens,
            )
        )
    dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = chunk_gated_delta_rule_bwd_qla(
        q,
        k,
        v,
        g_qla,
        beta,
        A_qla,
        do,
        dht,
        scale,
        h0,
        cu_seqlens,
    )

    if check_accuracy:
        if dq_fla is not None:
            print(
                f"dq_fla: {(dq_fla - dq_ref).abs().max().item():.4f} / {dq_ref.abs().max().item():.4f}"
            )
        print(
            f"dq_qla: {(dq_qla - dq_ref).abs().max().item():.4f} / {dq_ref.abs().max().item():.4f}"
        )
        if dk_fla is not None:
            print(
                f"dk_fla: {(dk_fla - dk_ref).abs().max().item():.4f} / {dk_ref.abs().max().item():.4f}"
            )
        print(
            f"dk_qla: {(dk_qla - dk_ref).abs().max().item():.4f} / {dk_ref.abs().max().item():.4f}"
        )
        if dv_fla is not None:
            print(
                f"dv_fla: {(dv_fla - dv_ref).abs().max().item():.4f} / {dv_ref.abs().max().item():.4f}"
            )
        print(
            f"dv_qla: {(dv_qla - dv_ref).abs().max().item():.4f} / {dv_ref.abs().max().item():.4f}"
        )
        if dht is not None and dh0_fla is not None:
            print(
                f"dh0_fla: {(dh0_fla - dh0_ref).abs().max().item():.4f} / {dh0_ref.abs().max().item():.4f}"
            )
        if dht is not None:
            print(
                f"dh0_qla: {(dh0_qla - dh0_ref).abs().max().item():.4f} / {dh0_ref.abs().max().item():.4f}"
            )
        if db_fla is not None:
            print(
                f"db_fla: {(db_fla - db_ref).abs().max().item():.4f} / {db_ref.abs().max().item():.4f}"
            )
        print(
            f"db_qla: {(db_qla - db_ref).abs().max().item():.4f} / {db_ref.abs().max().item():.4f}"
        )
        if dg_fla is not None:
            print(
                f"dg_fla: {(dg_fla - dg_ref).abs().max().item():.4f} / {dg_ref.abs().max().item():.4f}"
            )
        print(
            f"dg_qla: {(dg_qla - dg_ref).abs().max().item():.4f} / {dg_ref.abs().max().item():.4f}"
        )

        for repeat_idx in range(correctness_repeats):
            dq_qla, dk_qla, dv_qla, db_qla, dg_qla, dh0_qla = (
                chunk_gated_delta_rule_bwd_qla(
                    q,
                    k,
                    v,
                    g_qla,
                    beta,
                    A_qla,
                    do,
                    dht,
                    scale,
                    h0,
                    cu_seqlens,
                )
            )
            try:
                assert (
                    dq_qla - dq_ref
                ).abs().max().item() <= dq_ref.abs().max().item() * 0.02
                assert (
                    dk_qla - dk_ref
                ).abs().max().item() <= dk_ref.abs().max().item() * 0.02
                assert (
                    dv_qla - dv_ref
                ).abs().max().item() <= dv_ref.abs().max().item() * 0.02
                assert (
                    dg_qla - dg_ref
                ).abs().max().item() <= dg_ref.abs().max().item() * 0.02
                assert (
                    db_qla - db_ref
                ).abs().max().item() <= db_ref.abs().max().item() * 0.02
                if dht is not None:
                    assert (
                        dh0_qla - dh0_ref
                    ).abs().max().item() <= dh0_ref.abs().max().item() * 0.02
            except AssertionError as e:
                print("********** ERROR **********")
                print(f"bwd correctness repeat: {repeat_idx + 1}/{correctness_repeats}")
                print(
                    f"dq_qla: {(dq_qla - dq_ref).abs().max().item():.4f} / {dq_ref.abs().max().item():.4f}"
                )
                print(
                    f"dk_qla: {(dk_qla - dk_ref).abs().max().item():.4f} / {dk_ref.abs().max().item():.4f}"
                )
                print(
                    f"dv_qla: {(dv_qla - dv_ref).abs().max().item():.4f} / {dv_ref.abs().max().item():.4f}"
                )
                if dht is not None:
                    print(
                        f"dh0_qla: {(dh0_qla - dh0_ref).abs().max().item():.4f} / {dh0_ref.abs().max().item():.4f}"
                    )
                print(
                    f"db_qla: {(db_qla - db_ref).abs().max().item():.4f} / {db_ref.abs().max().item():.4f}"
                )
                print(
                    f"dg_qla: {(dg_qla - dg_ref).abs().max().item():.4f} / {dg_ref.abs().max().item():.4f}"
                )
                print("********** ERROR **********")
                raise e

    if show_speedup:
        prof_fla = None
        if not skip_fla_bwd:
            prof_fla = profile(
                chunk_gated_delta_rule_bwd_fla,
                [q, k, v, g_fla, beta, A_fla, scale, h0, do, dht, cu_seqlens],
            )
        prof_qla = profile(
            chunk_gated_delta_rule_bwd_qla,
            [q, k, v, g_qla, beta, A_qla, do, dht, scale, h0, cu_seqlens],
        )
        if prof_fla is not None:
            print(f"[bwd] prof_fla keys ({len(prof_fla)}): {sorted(prof_fla.keys())}")
        print(f"[bwd] prof_qla keys ({len(prof_qla)}): {sorted(prof_qla.keys())}")
        if prof_fla is not None:
            _print_profiler_note("[bwd] FLA", prof_fla)
        _print_profiler_note("[bwd] FlashQLA", prof_qla)
        result_fla = {}
        if prof_fla is not None:
            result_fla = {
                "[bwd] csum": _profile_value(
                    prof_fla, "[bwd] FLA csum", "chunk_local_cumsum_scalar_kernel"
                ),
                "[bwd] recom": _profile_value(
                    prof_fla,
                    "[bwd] FLA recom",
                    "recompute_w_u_fwd_kernel",
                    "chunk_gated_delta_rule_fwd_kernel_h_blockdim*",
                ),
                "[bwd] dv": _profile_value(
                    prof_fla, "[bwd] FLA dv", "chunk_bwd_kernel_dv_local"
                ),
                "[bwd] gdr": _profile_value(
                    prof_fla,
                    "[bwd] FLA gdr",
                    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim*",
                ),
                "[bwd] dqkwg": _profile_value(
                    prof_fla, "[bwd] FLA dqkwg", "kernel_kernel"
                ),
                "[bwd] wy": _profile_value(
                    prof_fla, "[bwd] FLA wy", "prepare_wy_repr_bwd_kernel"
                ),
            }
        result_qla = {
            "[bwd] csum": _profile_value(
                prof_qla,
                "[bwd] FlashQLA csum",
                "tilelang_chunk_local_cumsum_kernel_kernel",
            ),
            "[bwd] recom": _profile_value(
                prof_qla,
                "[bwd] FlashQLA recom",
                "tilelang_prepare_h_kernel_kernel",
            ),
            "[bwd] gdr": _profile_value(
                prof_qla,
                "[bwd] FlashQLA gdr",
                "tilelang_fused_chunk_gdr_bwd_kernel_kernel",
            ),
        }
        if num_k_heads < num_v_heads:
            if prof_fla is not None:
                result_fla["[bwd] reduc"] = _profile_value(
                    prof_fla, "[bwd] FLA reduc", "compress_heads_kernel"
                )
            result_qla["[bwd] reduc"] = _profile_value(
                prof_qla,
                "[bwd] FlashQLA reduc",
                "tilelang_group_reduce_vector_kernel_kernel",
                "tilelang_group_reduce_vector_kernel_kernel",
            )
        if prof_fla is not None:
            result_fla["total"] = prof_fla["total"]
        result_qla["total"] = prof_qla["total"]
        results = {"flash_qla": result_qla}
        if prof_fla is not None:
            results = {"fla": result_fla, **results}
        df = pd.DataFrame(results)
        print(df.round(3))
        if prof_fla is not None:
            speedup = results["fla"]["total"] / results["flash_qla"]["total"]
            print(f"Speed up: {speedup:2.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gated Delta Rule")
    parser.add_argument(
        "--set",
        type=str,
        default="develop",
        help="Preset name (loads from settings/{set}.csv)",
    )
    parser.add_argument(
        "--seqlen", "--num-tokens", type=int, default=16384, help="Sequence Length"
    )
    parser.add_argument(
        "--nkh",
        "--num-k-heads",
        type=int,
        default=0,
        help="Number of K heads (num_k_heads)",
    )
    parser.add_argument(
        "--nvh",
        "--num-heads",
        "--num-v-heads",
        type=int,
        default=64,
        help="Number of V heads (num_v_heads)",
    )
    parser.add_argument(
        "--no-h0",
        action="store_true",
        help="Disable initial state and gradient of final state",
    )
    parser.add_argument("--skip-bwd", action="store_true", help="Test forward only")
    parser.add_argument(
        "--no-cp",
        "--disable-auto-cp",
        action="store_true",
        help="Disable auto intra-card CP",
    )
    parser.add_argument(
        "--swa-ratio", type=float, default=0.75, help="Ratio of sliding-window heads"
    )
    parser.add_argument(
        "--data-dtype",
        type=str,
        default="bfloat16",
        help="Data type for input and output",
    )
    parser.add_argument(
        "--ref-dtype", type=str, default="float64", help="Data type for reference"
    )
    parser.add_argument("--hide-acc", action="store_true", help="Do not print accuracy")
    parser.add_argument("--hide-lat", action="store_true", help="Do not print latency")
    parser.add_argument(
        "--run-fla-bwd",
        action="store_true",
        help="Force running the FLA backward baseline even on Blackwell",
    )
    parser.add_argument(
        "--seed", "--random-seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    if args.nkh <= 0:
        args.nkh = args.nvh

    metadata = {
        "head_dim_k": 128,  # MUST BE 128
        "head_dim_v": 128,  # MUST BE 128
        "chunk_size": 64,  # MUST BE 64
        "num_tokens": args.seqlen,
        "num_k_heads": args.nkh,
        "num_v_heads": args.nvh,
        "use_h0": not args.no_h0,
        "data_dtype": args.data_dtype,
        "ref_dtype": args.ref_dtype,
        "check_accuracy": not args.hide_acc,
        "show_speedup": not args.hide_lat,
        "skip_bwd": args.skip_bwd,
        "skip_fla_bwd": False if args.run_fla_bwd else None,
        "auto_cp": not args.no_cp,
        "swa_ratio": args.swa_ratio,
        "random_seed": args.seed,
        "device": "cuda",
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    preset = pd.read_csv(os.path.join(script_dir, "settings", f"{args.set}.csv"))
    for i, row in preset.iterrows():
        print("-" * 64)
        torch.cuda.empty_cache()
        data = row.to_dict()
        if "cu_seqlens" in data.keys():
            data["cu_seqlens"] = list(map(int, data["cu_seqlens"].split("-")))
        metadata.update(data)
        test_gated_delta_rule(**metadata)
    print("-" * 64)
