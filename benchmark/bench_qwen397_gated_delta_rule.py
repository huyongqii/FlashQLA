#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Benchmark FlashQLA GDR forward on Qwen 397B/122B shapes.

This is intentionally closer to benchmark/bench_gated_delta_rule.py than to
tests/test_gdr.py: it creates inputs once per case and times direct function
calls with tilelang.profiler.do_bench(). Different FlashQLA policy envs are
still isolated in worker subprocesses because dispatch is decided at import
time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD_DIM = 128

SHAPES = {
    "tp1": (16, 64),
    "tp2": (8, 32),
    "tp4": (4, 16),
    "tp8": (2, 8),
}
SHAPE_GROUPS = {
    "qwen397": ("tp1", "tp2", "tp4", "tp8"),
    "all": ("tp1", "tp2", "tp4", "tp8"),
}
DEFAULT_SEQLENS = (4096, 8192, 16384, 32768)

POLICIES: dict[str, dict[str, str]] = {
    "hopper_compat": {
        "FLASHQLA_FORCE_ARCH": "sm90",
    },
    "qwen397_native": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h,o",
    },
    "qwen397_native_cp": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_CP": "1",
        "FLASHQLA_BLACKWELL_CP_DUAL_A": "1",
        "FLASHQLA_BLACKWELL_CP_SUMMARY_DTYPE": "bf16",
        "FLASHQLA_BLACKWELL_PREPARE_H_TCGEN05": "x",
        "FLASHQLA_CP_MAX_LOCAL_CHUNKS": "32",
        "FLASHQLA_CP_MIN_CHUNKS": "512",
        "FLASHQLA_CP_WARMUP_THRESHOLD": "-1.0",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h,o",
    },
    "qwen397_native_cp_force_s8": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_CP": "1",
        "FLASHQLA_BLACKWELL_CP_DUAL_A": "1",
        "FLASHQLA_BLACKWELL_CP_SUMMARY_DTYPE": "bf16",
        "FLASHQLA_BLACKWELL_PREPARE_H_TCGEN05": "x",
        "FLASHQLA_AUTOCP": "1",
        "FLASHQLA_CP_MAX_LOCAL_CHUNKS": "8",
        "FLASHQLA_CP_MIN_CHUNKS": "1",
        "FLASHQLA_CP_WARMUP_THRESHOLD": "-1.0",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h,o",
    },
}

ENV_TO_CLEAR = (
    "FLASHQLA_FORCE_ARCH",
    "FLASHQLA_SUPPRESS_BLACKWELL_WARNING",
    "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE",
    "FLASHQLA_BLACKWELL_NATIVE",
    "FLASHQLA_BLACKWELL_NATIVE_KERNELS",
    "FLASHQLA_BLACKWELL_BLOCK_DV",
    "FLASHQLA_BLACKWELL_FWD_THREADS",
    "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS",
    "FLASHQLA_BLACKWELL_FWD_POLICY",
    "FLASHQLA_BLACKWELL_CP",
    "FLASHQLA_BLACKWELL_CP_EXACT",
    "FLASHQLA_BLACKWELL_CP_DUAL_A",
    "FLASHQLA_BLACKWELL_CP_SUMMARY_DTYPE",
    "FLASHQLA_BLACKWELL_CP_START_FIX_TL",
    "FLASHQLA_BLACKWELL_PREPARE_H_V2",
    "FLASHQLA_BLACKWELL_PREPARE_H_TCGEN05",
    "FLASHQLA_BLACKWELL_PRETRANSFORM_A",
    "FLASHQLA_CP_EXACT",
    "FLASHQLA_AUTOCP",
    "FLASHQLA_CP_MAX_LOCAL_CHUNKS",
    "FLASHQLA_CP_MIN_CHUNKS",
    "FLASHQLA_CP_WARMUP_THRESHOLD",
    "FLASHQLA_CP_CORRECT_H0_TORCH",
    "FLASHQLA_BLOCK_DV",
    "FLASHQLA_TARGET_CTA_RATIO",
)

CSV_FIELDS = (
    "policy",
    "shape",
    "batch_size",
    "seqlen",
    "nkh",
    "nvh",
    "status",
    "qla_ms",
    "fla_ms",
    "fi_ms",
    "speedup_vs_fla",
    "speedup_vs_fi",
    "check_o_rel",
    "check_s_rel",
    "error",
)


@dataclass
class Row:
    policy: str
    shape: str
    batch_size: int
    seqlen: int
    nkh: int
    nvh: int
    status: str
    qla_ms: float
    fla_ms: float
    fi_ms: float
    speedup_vs_fla: float
    speedup_vs_fi: float
    check_o_rel: float
    check_s_rel: float
    error: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Row":
        return cls(
            policy=str(data.get("policy", "")),
            shape=str(data.get("shape", "")),
            batch_size=int(data.get("batch_size", 1)),
            seqlen=int(data.get("seqlen", 0)),
            nkh=int(data.get("nkh", 0)),
            nvh=int(data.get("nvh", 0)),
            status=str(data.get("status", "failed")),
            qla_ms=_as_float(data.get("qla_ms")),
            fla_ms=_as_float(data.get("fla_ms")),
            fi_ms=_as_float(data.get("fi_ms")),
            speedup_vs_fla=_as_float(data.get("speedup_vs_fla")),
            speedup_vs_fi=_as_float(data.get("speedup_vs_fi")),
            check_o_rel=_as_float(data.get("check_o_rel")),
            check_s_rel=_as_float(data.get("check_s_rel")),
            error=str(data.get("error", "")),
        )

    def to_csv(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "shape": self.shape,
            "batch_size": self.batch_size,
            "seqlen": self.seqlen,
            "nkh": self.nkh,
            "nvh": self.nvh,
            "status": self.status,
            "qla_ms": _csv_float(self.qla_ms),
            "fla_ms": _csv_float(self.fla_ms),
            "fi_ms": _csv_float(self.fi_ms),
            "speedup_vs_fla": _csv_float(self.speedup_vs_fla),
            "speedup_vs_fi": _csv_float(self.speedup_vs_fi),
            "check_o_rel": _csv_float(self.check_o_rel),
            "check_s_rel": _csv_float(self.check_s_rel),
            "error": self.error,
        }


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _csv_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _fmt_ms(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.3f}ms"


def _fmt_x(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.3f}x"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in _split_csv(value)]


def _expand_shapes(items: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in items:
        if item in SHAPE_GROUPS:
            expanded.extend(SHAPE_GROUPS[item])
        else:
            expanded.append(item)
    return expanded


def _parse_env(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env has empty key in {item!r}")
        parsed[key] = value
    return parsed


def _policy_env(policy: str, extra_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key in ENV_TO_CLEAR:
        env.pop(key, None)
    env.update(POLICIES[policy])
    env.update(extra_env)
    env.setdefault("FLASHQLA_SUPPRESS_BLACKWELL_WARNING", "1")
    return env


def _speedup(base_ms: float, qla_ms: float) -> float:
    if math.isnan(base_ms) or math.isnan(qla_ms) or qla_ms <= 0:
        return float("nan")
    return base_ms / qla_ms


def _mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _worker_json(row: dict[str, Any]) -> int:
    print(json.dumps(row, sort_keys=True), flush=True)
    return 0


def _worker_failed(args: argparse.Namespace, status: str, error: str) -> int:
    return _worker_json(
        {
            "policy": args.worker_policy,
            "shape": args.worker_shape,
            "batch_size": 1,
            "seqlen": args.worker_seqlen,
            "nkh": args.worker_nkh,
            "nvh": args.worker_nvh,
            "status": status,
            "qla_ms": None,
            "fla_ms": None,
            "fi_ms": None,
            "speedup_vs_fla": None,
            "speedup_vs_fi": None,
            "check_o_rel": None,
            "check_s_rel": None,
            "error": error[-2000:],
        }
    )


def _run_worker(args: argparse.Namespace) -> int:
    try:
        sys.path.insert(0, str(REPO_ROOT))

        import torch
        import torch.nn.functional as F
        import tilelang

        from flash_qla import chunk_gated_delta_rule_fwd as qla_fwd
        from flash_qla.utils import l2norm

        if not torch.cuda.is_available():
            return _worker_failed(args, "failed", "CUDA not available")

        try:
            from fla.ops.gated_delta_rule.chunk import (
                chunk_gated_delta_rule_fwd as fla_fwd,
            )
        except Exception as exc:
            if args.skip_fla:
                fla_fwd = None
            else:
                return _worker_failed(args, "failed", f"FLA import failed: {exc}")

        fi_fwd = None
        if not args.skip_fi:
            try:
                from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_fwd
            except Exception:
                fi_fwd = None

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        device = "cuda"
        batch_size = 1
        t = args.worker_seqlen
        hq = args.worker_nkh
        hv = args.worker_nvh
        scale = HEAD_DIM ** (-0.5)
        dtype = torch.bfloat16

        q = l2norm(torch.randn(batch_size, t, hq, HEAD_DIM, device=device, dtype=dtype))
        k = l2norm(torch.randn(batch_size, t, hq, HEAD_DIM, device=device, dtype=dtype))
        v = torch.randn(batch_size, t, hv, HEAD_DIM, device=device, dtype=dtype)
        g = F.logsigmoid(
            torch.randn(batch_size, t, hv, device=device, dtype=torch.float32)
        ) / 16
        beta = torch.randn(batch_size, t, hv, device=device, dtype=torch.float32).sigmoid()
        h0 = None
        if not args.no_h0:
            h0 = torch.randn(
                batch_size, hv, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32
            )

        swa_mask = torch.zeros(hv, dtype=torch.bool, device=device)
        swa_mask[: math.ceil(args.swa_ratio * hv)] = True
        swa_mask = swa_mask[torch.randperm(hv, device=device)]
        g[:, :, ~swa_mask] = 0.0

        fixed_cu_seqlens = torch.tensor([0, t], dtype=torch.int32, device=device)

        def call_qla():
            return qla_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                output_h=False,
                cu_seqlens=None,
                auto_cp=not args.no_cp,
            )

        def call_fla():
            assert fla_fwd is not None
            return fla_fwd(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=None,
            )

        def call_fi():
            assert fi_fwd is not None
            return fi_fwd(
                q=q.view(-1, hq, HEAD_DIM),
                k=k.view(-1, hq, HEAD_DIM),
                v=v.view(-1, hv, HEAD_DIM),
                g=g.view(-1, hv),
                beta=beta.view(-1, hv),
                scale=scale,
                initial_state=h0,
                cu_seqlens=fixed_cu_seqlens,
                output_final_state=True,
            )

        check_o_rel = float("nan")
        check_s_rel = float("nan")
        if args.check:
            if fla_fwd is None:
                return _worker_failed(args, "failed", "Correctness check needs FLA")
            qla_result = call_qla()
            fla_result = call_fla()
            _, _, o_qla, _, s_qla = qla_result
            _, o_fla, _, s_fla, _, _ = fla_result
            o_abs = (o_qla.float() - o_fla.float()).abs().amax().item()
            s_abs = (s_qla.float() - s_fla.float()).abs().amax().item()
            o_denom = o_fla.float().abs().amax().clamp_min(1e-6).item()
            s_denom = s_fla.float().abs().amax().clamp_min(1e-6).item()
            check_o_rel = o_abs / o_denom
            check_s_rel = s_abs / s_denom
            if check_o_rel > args.rtol or check_s_rel > args.rtol:
                return _worker_json(
                    {
                        "policy": args.worker_policy,
                        "shape": args.worker_shape,
                        "batch_size": batch_size,
                        "seqlen": t,
                        "nkh": hq,
                        "nvh": hv,
                        "status": "failed_check",
                        "qla_ms": None,
                        "fla_ms": None,
                        "fi_ms": None,
                        "speedup_vs_fla": None,
                        "speedup_vs_fi": None,
                        "check_o_rel": check_o_rel,
                        "check_s_rel": check_s_rel,
                        "error": (
                            f"check failed: o_rel={check_o_rel:.6f}, "
                            f"s_rel={check_s_rel:.6f}, rtol={args.rtol}"
                        ),
                    }
                )

        torch.cuda.synchronize()
        qla_ms = tilelang.profiler.do_bench(
            call_qla, warmup=args.warmup, rep=args.repeats
        )

        fla_ms = float("nan")
        if fla_fwd is not None and not args.skip_fla:
            torch.cuda.synchronize()
            fla_ms = tilelang.profiler.do_bench(
                call_fla, warmup=args.warmup, rep=args.repeats
            )

        fi_ms = float("nan")
        if fi_fwd is not None and not args.skip_fi:
            torch.cuda.synchronize()
            fi_ms = tilelang.profiler.do_bench(
                call_fi, warmup=args.warmup, rep=args.repeats
            )

        return _worker_json(
            {
                "policy": args.worker_policy,
                "shape": args.worker_shape,
                "batch_size": batch_size,
                "seqlen": t,
                "nkh": hq,
                "nvh": hv,
                "status": "ok",
                "qla_ms": qla_ms,
                "fla_ms": fla_ms,
                "fi_ms": fi_ms,
                "speedup_vs_fla": _speedup(fla_ms, qla_ms),
                "speedup_vs_fi": _speedup(fi_ms, qla_ms),
                "check_o_rel": check_o_rel,
                "check_s_rel": check_s_rel,
                "error": "",
            }
        )
    except Exception:
        return _worker_failed(args, "failed", traceback.format_exc())


def _parse_worker_output(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _run_case(
    *,
    policy: str,
    shape: str,
    seqlen: int,
    nkh: int,
    nvh: int,
    args: argparse.Namespace,
    log_dir: Path,
    extra_env: dict[str, str],
) -> Row:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-policy",
        policy,
        "--worker-shape",
        shape,
        "--worker-seqlen",
        str(seqlen),
        "--worker-nkh",
        str(nkh),
        "--worker-nvh",
        str(nvh),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--swa-ratio",
        str(args.swa_ratio),
        "--rtol",
        str(args.rtol),
    ]
    if args.no_h0:
        cmd.append("--no-h0")
    if args.no_cp:
        cmd.append("--no-cp")
    if not args.check:
        cmd.append("--no-check")
    if args.skip_fla:
        cmd.append("--skip-fla")
    if args.skip_fi:
        cmd.append("--skip-fi")

    env = _policy_env(policy, extra_env)
    visible_env = {key: env[key] for key in sorted(POLICIES[policy]) if key in env}
    for key in sorted(extra_env):
        visible_env[key] = env[key]
    log_path = log_dir / f"{policy}_{shape}_t{seqlen}.log"

    header = [
        "cmd=" + " ".join(cmd),
        "env=" + json.dumps(visible_env, sort_keys=True),
        "",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
            check=False,
        )
        log_path.write_text(
            "\n".join(header)
            + "STDOUT\n"
            + proc.stdout
            + "\nSTDERR\n"
            + proc.stderr,
            encoding="utf-8",
        )
        data = _parse_worker_output(proc.stdout)
        if data is None:
            return Row(
                policy,
                shape,
                1,
                seqlen,
                nkh,
                nvh,
                "failed",
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                f"worker rc={proc.returncode}; no JSON output; see {log_path}",
            )
        if proc.returncode != 0 and data.get("status") == "ok":
            data["status"] = "failed"
            data["error"] = f"worker rc={proc.returncode}; see {log_path}"
        return Row.from_dict(data)
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(
            "\n".join(header) + f"TIMEOUT after {args.timeout}s\n{exc}",
            encoding="utf-8",
        )
        return Row(
            policy,
            shape,
            1,
            seqlen,
            nkh,
            nvh,
            "timeout",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            f"timeout after {args.timeout}s; see {log_path}",
        )


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv())


def _build_summary(rows: list[Row]) -> str:
    grouped: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        grouped.setdefault((row.policy, row.shape), []).append(row)

    aggregate_rows: list[list[str]] = []
    for (policy, shape), group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if row.status == "ok"]
        speedups = [row.speedup_vs_fla for row in ok_rows]
        aggregate_rows.append(
            [
                policy,
                shape,
                f"{len(ok_rows)}/{len(group_rows)}",
                _fmt_x(_mean(speedups)),
                _fmt_x(min(speedups)) if speedups else "",
                _fmt_x(max(speedups)) if speedups else "",
                _fmt_ms(_mean([row.qla_ms for row in ok_rows])),
                _fmt_ms(_mean([row.fla_ms for row in ok_rows])),
                _fmt_ms(_mean([row.fi_ms for row in ok_rows])),
            ]
        )

    best_rows: list[list[str]] = []
    shape_order = [
        shape for shape in SHAPE_GROUPS["qwen397"] if any(row.shape == shape for row in rows)
    ]
    seqlen_order = sorted({row.seqlen for row in rows})
    for shape in shape_order:
        for seqlen in seqlen_order:
            candidates = [
                row
                for row in rows
                if row.shape == shape and row.seqlen == seqlen and row.status == "ok"
            ]
            candidates.sort(key=lambda row: row.speedup_vs_fla, reverse=True)
            best = candidates[0] if candidates else None
            next_best = candidates[1] if len(candidates) > 1 else None
            best_rows.append(
                [
                    shape,
                    "1",
                    str(seqlen),
                    best.policy if best else "",
                    _fmt_x(best.speedup_vs_fla) if best else "",
                    next_best.policy if next_best else "",
                    _fmt_x(next_best.speedup_vs_fla) if next_best else "",
                ]
            )

    failed_rows = [
        [
            row.policy,
            row.shape,
            str(row.seqlen),
            row.status,
            row.error.replace("\n", " | ")[:240],
        ]
        for row in rows
        if row.status != "ok"
    ]

    parts = [
        "Aggregate",
        "",
        _markdown_table(
            ["policy", "shape", "ok", "avg", "min", "max", "qla", "fla", "fi"],
            aggregate_rows,
        ),
        "",
        "Best by case",
        "",
        _markdown_table(
            ["shape", "B", "T", "best_policy", "best", "next_policy", "next"],
            best_rows,
        ),
    ]
    if failed_rows:
        parts.extend(
            [
                "",
                "Failures",
                "",
                _markdown_table(["policy", "shape", "T", "status", "error"], failed_rows),
            ]
        )
    return "\n".join(parts)


def _print_progress(row: Row) -> None:
    if row.status == "ok":
        print(
            f"{row.policy:28s} {row.shape:4s} T={row.seqlen:<5d} "
            f"qla={_fmt_ms(row.qla_ms):>10s} fla={_fmt_ms(row.fla_ms):>10s} "
            f"vs_fla={_fmt_x(row.speedup_vs_fla):>8s}",
            flush=True,
        )
    else:
        print(
            f"{row.policy:28s} {row.shape:4s} T={row.seqlen:<5d} "
            f"status={row.status} error={row.error[:120]}",
            flush=True,
        )


def _run_driver(args: argparse.Namespace) -> int:
    selected_policies = _split_csv(args.policies)
    if args.with_hopper and "hopper_compat" not in selected_policies:
        selected_policies.append("hopper_compat")
    selected_shapes = _expand_shapes(_split_csv(args.shapes))
    seqlens = _parse_int_csv(args.seqlens)
    extra_env = _parse_env(args.env)

    unknown_policies = [policy for policy in selected_policies if policy not in POLICIES]
    unknown_shapes = [shape for shape in selected_shapes if shape not in SHAPES]
    if unknown_policies:
        raise ValueError(f"Unknown policies: {unknown_policies}. Valid: {sorted(POLICIES)}")
    if unknown_shapes:
        raise ValueError(f"Unknown shapes: {unknown_shapes}. Valid: {sorted(SHAPES)}")
    if any(seqlen <= 0 for seqlen in seqlens):
        raise ValueError("--seqlens entries must be positive")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Qwen 397B/122B forward benchmark")
    print(f"policies={','.join(selected_policies)}")
    print(f"shapes={','.join(selected_shapes)} seqlens={','.join(map(str, seqlens))}")
    print(
        f"warmup={args.warmup} repeats={args.repeats} "
        f"check={args.check} h0={not args.no_h0} auto_cp={not args.no_cp}",
        flush=True,
    )

    rows: list[Row] = []
    for policy in selected_policies:
        for shape in selected_shapes:
            nkh, nvh = SHAPES[shape]
            for seqlen in seqlens:
                row = _run_case(
                    policy=policy,
                    shape=shape,
                    seqlen=seqlen,
                    nkh=nkh,
                    nvh=nvh,
                    args=args,
                    log_dir=log_dir,
                    extra_env=extra_env,
                )
                rows.append(row)
                _print_progress(row)

    out_path = Path(args.out)
    _write_csv(out_path, rows)
    summary = _build_summary(rows)
    summary_path = out_path.with_suffix(".summary.md")
    summary_path.write_text(summary + "\n", encoding="utf-8")

    print("\n" + summary)
    print(f"\nwrote {out_path}")
    print(f"wrote {summary_path}")
    return 1 if any(row.status != "ok" for row in rows) else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies",
        default="qwen397_native_cp",
        help=f"Comma-separated policies. Available: {','.join(sorted(POLICIES))}",
    )
    parser.add_argument(
        "--with-hopper",
        action="store_true",
        help="Also run hopper_compat.",
    )
    parser.add_argument(
        "--shapes",
        default="qwen397",
        help=(
            "Comma-separated shape presets or groups. "
            f"Shapes: {','.join(sorted(SHAPES))}. "
            f"Groups: {','.join(sorted(SHAPE_GROUPS))}"
        ),
    )
    parser.add_argument(
        "--seqlens",
        default=",".join(str(x) for x in DEFAULT_SEQLENS),
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", default="qwen397_gdr_bench.csv")
    parser.add_argument("--log-dir", default="qwen397_gdr_bench_logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swa-ratio", type=float, default=0.75)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--no-h0", action="store_true")
    parser.add_argument("--no-cp", action="store_true")
    parser.add_argument("--no-check", action="store_false", dest="check")
    parser.set_defaults(check=True)
    parser.add_argument("--skip-fla", action="store_true")
    parser.add_argument("--skip-fi", action="store_true")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra worker env override, e.g. --env FLASHQLA_BLOCK_DV=64.",
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-policy", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-shape", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seqlen", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-nkh", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-nvh", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.worker:
        return _run_worker(args)
    return _run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
