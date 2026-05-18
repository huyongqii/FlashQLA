# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Debug Blackwell segmented-forward initial states.

This isolates the segmented experiment from the benchmark tail.  It compares the
state produced by the segment-prefix preparation path against the reference
chunk state at each segment boundary.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from ref_gdr import chunk_gated_delta_rule_fwd as ref_fwd  # noqa: E402
from flash_qla.ops.gated_delta_rule.chunk import (  # noqa: E402
    correct_initial_states,
    fused_gdr_h,
    kkt_solve,
)
from flash_qla.ops.utils import chunk_local_cumsum  # noqa: E402
from flash_qla.utils import l2norm  # noqa: E402


def _max_report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max()
    idx = tuple(int(i) for i in torch.unravel_index(diff.argmax(), diff.shape))
    denom = expected.float().abs().max().clamp_min(1e-12)
    print(
        f"{name}: max_abs={max_abs.item():.6f} "
        f"rel={(max_abs / denom).item():.6f} idx={idx} "
        f"actual={actual[idx].float().item():.6f} "
        f"expected={expected[idx].float().item():.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--nkh", type=int, required=True)
    parser.add_argument("--nvh", type=int, required=True)
    parser.add_argument("--segment-chunks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swa-ratio", type=float, default=0.75)
    parser.add_argument("--no-h0", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 1
    chunk_size = 64
    head_dim = 128
    segment_tokens = args.segment_chunks * chunk_size
    assert args.tokens % segment_tokens == 0
    num_segments = args.tokens // segment_tokens

    q = l2norm(torch.randn((batch_size, args.tokens, args.nkh, head_dim), device=device, dtype=dtype))
    k = l2norm(torch.randn((batch_size, args.tokens, args.nkh, head_dim), device=device, dtype=dtype))
    v = torch.randn((batch_size, args.tokens, args.nvh, head_dim), device=device, dtype=dtype)
    g_raw = torch.nn.functional.logsigmoid(
        torch.randn((batch_size, args.tokens, args.nvh), device=device, dtype=torch.float32)
    ) / 16
    beta = torch.randn((batch_size, args.tokens, args.nvh), device=device, dtype=torch.float32).sigmoid()
    h0 = None
    if not args.no_h0:
        h0 = torch.randn((batch_size, args.nvh, head_dim, head_dim), device=device, dtype=torch.float32)

    swa_mask = torch.zeros((args.nvh), dtype=torch.bool, device=device)
    swa_mask[: math.ceil(args.swa_ratio * args.nvh)] = 1
    swa_mask = swa_mask[torch.randperm(args.nvh, device=device)]
    g_raw[:, :, ~swa_mask] = 0.0

    print(
        f"device={torch.cuda.get_device_name()} sm_{torch.cuda.get_device_capability()[0]}"
        f"{torch.cuda.get_device_capability()[1]}"
    )
    print(
        f"shape=B1 T{args.tokens} Hg{args.nkh} H{args.nvh} "
        f"segment_chunks={args.segment_chunks} segments={num_segments}"
    )

    scale = head_dim ** -0.5
    g_ref, _o_ref, _A_ref, h_ref, final_ref = ref_fwd(
        q=q.to(torch.float64),
        k=k.to(torch.float64),
        v=v.to(torch.float64),
        g=g_raw.to(torch.float64),
        beta=beta.to(torch.float64),
        scale=scale,
        initial_state=h0,
    )
    g_qla = chunk_local_cumsum(g_raw, chunk_size=chunk_size)
    A_qla = kkt_solve(k=k, b=beta, g=g_qla, cu_seqlens=None)

    seqlen_dtype = torch.int32
    cu_segments = (
        torch.arange(num_segments + 1, dtype=seqlen_dtype, device=device)
        * segment_tokens
    )
    num_warmup_chunks = torch.full(
        (num_segments, args.nvh),
        args.segment_chunks,
        dtype=seqlen_dtype,
        device=device,
    )
    fallback_mask = torch.ones((num_segments, args.nvh), dtype=torch.bool, device=device)
    _h, ht, mt = fused_gdr_h(
        k=k,
        v=v,
        a=A_qla,
        g=g_qla,
        b=beta,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cu_segments,
        num_warmup_chunks=num_warmup_chunks,
    )
    seq_map_r2c = torch.tensor([0, num_segments], dtype=seqlen_dtype, device=device)
    segment_h0 = correct_initial_states(
        raw_h0=h0,
        ht_buffer=ht,
        mt_buffer=mt,
        fallback_mask=fallback_mask,
        seq_map_r2c=seq_map_r2c,
    )

    if h0 is not None:
        _max_report("segment_h0[0]_vs_raw_h0", segment_h0[0], h0[0])
    for seg in range(num_segments):
        chunk_idx = seg * args.segment_chunks
        _max_report(
            f"segment_h0[{seg}]_vs_ref_h_chunk{chunk_idx}",
            segment_h0[seg],
            h_ref[0, chunk_idx],
        )
    _max_report("last_segment_final_vs_ref_final", ht[-1].float(), final_ref[0].float())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
