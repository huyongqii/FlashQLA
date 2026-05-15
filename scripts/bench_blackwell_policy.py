#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Sweep stable Blackwell FlashQLA GDR forward policies.

This script intentionally keeps the default sweep to the stable native forward
path. Risky experiments such as pipeline, dv128_reuse, pg_precompute, and
tmem_v2 should be tested separately because they have shown correctness or
timeout failures on B200/B300.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SHAPES = {
    "h64": (64, 64),
    "tp8": (2, 8),
    "tp4": (4, 16),
    "tp2": (8, 32),
    "tp1": (16, 64),
}


SHAPE_GROUPS = {
    "qwen397": ("tp8", "tp4", "tp2", "tp1"),
    "all": tuple(SHAPES),
}


POLICIES = {
    "compat": {},
    "auto_256_lh": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "auto",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h",
    },
    "qwen397_native": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h",
    },
    "qwen397_small_hv": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_FWD_EXPERIMENT": "small_hv",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h",
    },
    "qwen397_small_hv_recompute": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_FWD_EXPERIMENT": "small_hv",
        "FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P": "1",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h",
    },
    "ag_256_lh": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h",
    },
    "ag_256_lho": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h,o",
    },
    "ag_256_lhag": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h,ag",
    },
    "ag_128_lh": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "64",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "128",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h",
    },
    "ag_b128_256_lh": {
        "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd,kkt",
        "FLASHQLA_BLACKWELL_FWD_POLICY": "native",
        "FLASHQLA_BLACKWELL_BLOCK_DV": "128",
        "FLASHQLA_BLACKWELL_FWD_THREADS": "256",
        "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS": "load,h",
    },
}


ENV_TO_CLEAR = (
    "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE",
    "FLASHQLA_BLACKWELL_NATIVE",
    "FLASHQLA_BLACKWELL_NATIVE_KERNELS",
    "FLASHQLA_BLACKWELL_BLOCK_DV",
    "FLASHQLA_BLACKWELL_FWD_THREADS",
    "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS",
    "FLASHQLA_BLACKWELL_FWD_POLICY",
    "FLASHQLA_BLACKWELL_FWD_EXPERIMENT",
    "FLASHQLA_BLACKWELL_FWD_MAX_ITERS",
    "FLASHQLA_BLACKWELL_KKT_EXPERIMENT",
    "FLASHQLA_BLACKWELL_PRECOMPUTE_P",
    "FLASHQLA_BLACKWELL_PRETRANSFORM_A",
    "FLASHQLA_BLACKWELL_SMALL_HV_RECOMPUTE_P",
    "FLASHQLA_CORRECTNESS_REPEATS",
)


SHAPE_RE = re.compile(r"Shape: B=(?P<b>\d+) Hk=(?P<hk>\d+) Hv=(?P<hv>\d+) T=(?P<t>\d+)")
TOTAL_RE = re.compile(r"^total\s+(?P<fla>[0-9.]+|NaN)\s+(?P<qla>[0-9.]+|NaN)\s*$")
SPEEDUP_RE = re.compile(r"Speed up:\s+(?P<speedup>[0-9.]+)x")
KERNEL_RE = re.compile(r"'(?P<kernel>tilelang_[^']+_kernel)'")


@dataclass
class RunResult:
    policy: str
    shape: str
    nkh: int
    nvh: int
    t: int | None
    fla_ms: float | None
    qla_ms: float | None
    speedup: float | None
    status: str
    returncode: int | None
    error_tail: str
    log_path: str
    kernels: str


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _expand_shapes(items: list[str]) -> list[str]:
    expanded: list[str] = []
    for item in items:
        if item in SHAPE_GROUPS:
            expanded.extend(SHAPE_GROUPS[item])
        else:
            expanded.append(item)
    return list(dict.fromkeys(expanded))


def _parse_float(value: str) -> float | None:
    if value == "NaN":
        return None
    return float(value)


def _parse_output(text: str) -> tuple[list[dict[str, float | int | None]], list[str]]:
    rows: list[dict[str, float | int | None]] = []
    kernels: set[str] = set()
    current_t: int | None = None
    pending_total: tuple[float | None, float | None] | None = None

    for line in text.splitlines():
        shape_match = SHAPE_RE.search(line)
        if shape_match:
            current_t = int(shape_match.group("t"))
            pending_total = None
            continue

        for kernel_match in KERNEL_RE.finditer(line):
            kernels.add(kernel_match.group("kernel"))

        total_match = TOTAL_RE.match(line.strip())
        if total_match:
            pending_total = (
                _parse_float(total_match.group("fla")),
                _parse_float(total_match.group("qla")),
            )
            continue

        speedup_match = SPEEDUP_RE.search(line)
        if speedup_match and current_t is not None and pending_total is not None:
            rows.append(
                {
                    "t": current_t,
                    "fla_ms": pending_total[0],
                    "qla_ms": pending_total[1],
                    "speedup": float(speedup_match.group("speedup")),
                }
            )
            pending_total = None

    return rows, sorted(kernels)


def _error_tail(text: str, max_lines: int = 12) -> str:
    interesting = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and (
            "Traceback" in line
            or "Error" in line
            or "ERROR" in line
            or "Exception" in line
            or "AssertionError" in line
            or "RuntimeError" in line
            or "ValueError" in line
            or "timeout" in line.lower()
        )
    ]
    tail = interesting[-max_lines:]
    return " | ".join(tail)


def _make_env(policy: str, correctness_repeats: int | None) -> dict[str, str]:
    env = os.environ.copy()
    for key in ENV_TO_CLEAR:
        env.pop(key, None)
    env.update(POLICIES[policy])
    if correctness_repeats is not None:
        env["FLASHQLA_CORRECTNESS_REPEATS"] = str(correctness_repeats)
    return env


def _run_one(
    *,
    policy: str,
    shape: str,
    nkh: int,
    nvh: int,
    args: argparse.Namespace,
    log_dir: Path,
) -> list[RunResult]:
    env = _make_env(policy, args.correctness_repeats)
    cmd = [
        sys.executable,
        "tests/test_gdr.py",
        "--set",
        args.set,
        "--skip-bwd",
        "--nkh",
        str(nkh),
        "--nvh",
        str(nvh),
    ]
    if args.no_cp:
        cmd.append("--no-cp")
    if args.hide_acc:
        cmd.append("--hide-acc")
    if args.hide_lat:
        cmd.append("--hide-lat")

    log_path = log_dir / f"{policy}_{shape}.log"
    print("=" * 80, flush=True)
    print(f"policy={policy} shape={shape} Hk={nkh} Hv={nvh}", flush=True)
    print("cmd=" + " ".join(cmd), flush=True)
    visible_env = {key: env[key] for key in sorted(POLICIES[policy])}
    if args.correctness_repeats is not None:
        visible_env["FLASHQLA_CORRECTNESS_REPEATS"] = str(args.correctness_repeats)
    print(f"env={visible_env}", flush=True)

    proc: subprocess.Popen[str] | None = None
    status = "ok"
    returncode: int | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        stdout, _ = proc.communicate(timeout=args.timeout)
        returncode = proc.returncode
        if returncode != 0:
            status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
        if proc is not None:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, _ = proc.communicate()
            returncode = proc.returncode

    log_path.write_text(stdout, encoding="utf-8")
    parsed, kernels = _parse_output(stdout)
    kernel_text = ";".join(kernels)
    error_tail = _error_tail(stdout) if status != "ok" else ""

    if not parsed:
        return [
            RunResult(
                policy=policy,
                shape=shape,
                nkh=nkh,
                nvh=nvh,
                t=None,
                fla_ms=None,
                qla_ms=None,
                speedup=None,
                status=status,
                returncode=returncode,
                error_tail=error_tail,
                log_path=str(log_path),
                kernels=kernel_text,
            )
        ]

    return [
        RunResult(
            policy=policy,
            shape=shape,
            nkh=nkh,
            nvh=nvh,
            t=int(row["t"]),
            fla_ms=row["fla_ms"],
            qla_ms=row["qla_ms"],
            speedup=row["speedup"],
            status=status,
            returncode=returncode,
            error_tail=error_tail,
            log_path=str(log_path),
            kernels=kernel_text,
        )
        for row in parsed
    ]


def _write_csv(path: Path, rows: list[RunResult]) -> None:
    fieldnames = (
        "policy",
        "shape",
        "nkh",
        "nvh",
        "t",
        "fla_ms",
        "qla_ms",
        "speedup",
        "status",
        "returncode",
        "error_tail",
        "kernels",
        "log_path",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_summary(rows: list[RunResult]) -> None:
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for row in rows:
        grouped.setdefault((row.policy, row.shape), []).append(row)

    print("\nSummary", flush=True)
    for (policy, shape), group in sorted(grouped.items()):
        speedups = [row.speedup for row in group if row.speedup is not None]
        if not speedups:
            status = ",".join(sorted({row.status for row in group}))
            print(f"{policy:16s} {shape:6s} status={status}", flush=True)
            continue
        avg = sum(speedups) / len(speedups)
        best = max(speedups)
        worst = min(speedups)
        print(
            f"{policy:16s} {shape:6s} avg={avg:.3f}x "
            f"min={worst:.3f}x max={best:.3f}x",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies",
        default="qwen397_native,auto_256_lh,compat",
        help=f"Comma-separated policies. Available: {','.join(sorted(POLICIES))}",
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
    parser.add_argument("--set", default="profile")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--correctness-repeats", type=int, default=1000)
    parser.add_argument("--out", default="blackwell_policy.csv")
    parser.add_argument("--log-dir", default="blackwell_policy_logs")
    parser.add_argument("--no-cp", action="store_true", default=True)
    parser.add_argument("--with-cp", action="store_false", dest="no_cp")
    parser.add_argument("--hide-acc", action="store_true")
    parser.add_argument("--hide-lat", action="store_true")
    args = parser.parse_args()

    selected_policies = _split_csv(args.policies)
    selected_shapes = _expand_shapes(_split_csv(args.shapes))
    unknown_policies = [item for item in selected_policies if item not in POLICIES]
    unknown_shapes = [item for item in selected_shapes if item not in SHAPES]
    if unknown_policies:
        raise ValueError(f"Unknown policies: {unknown_policies}. Valid: {sorted(POLICIES)}")
    if unknown_shapes:
        raise ValueError(f"Unknown shapes: {unknown_shapes}. Valid: {sorted(SHAPES)}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[RunResult] = []
    for policy in selected_policies:
        for shape in selected_shapes:
            nkh, nvh = SHAPES[shape]
            rows.extend(
                _run_one(
                    policy=policy,
                    shape=shape,
                    nkh=nkh,
                    nvh=nvh,
                    args=args,
                    log_dir=log_dir,
                )
            )

    out_path = Path(args.out)
    _write_csv(out_path, rows)
    _print_summary(rows)
    print(f"\nwrote {out_path}", flush=True)
    return 1 if any(row.status != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
