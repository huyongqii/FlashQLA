#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Sweep stable Blackwell FlashQLA GDR forward policies.

This script intentionally keeps the default sweep to the stable native forward
path. Failed Blackwell experiments are kept out of the policy table so benchmark
summaries stay focused on the supported path and simple tuning variants.
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
    'auto_256_lh': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'auto',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_safe': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o,hscale',
    },
    'qwen397_native_auto_dv': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_exact_tmem': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_TMEM_WIDTH': 'exact',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_cp': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_CP': '1',
        'FLASHQLA_CP_MAX_LOCAL_CHUNKS': '32',
        'FLASHQLA_CP_MIN_CHUNKS': '512',
        'FLASHQLA_CP_WARMUP_THRESHOLD': '-10.0',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_cp_s16': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_CP': '1',
        'FLASHQLA_CP_MAX_LOCAL_CHUNKS': '16',
        'FLASHQLA_CP_MIN_CHUNKS': '512',
        'FLASHQLA_CP_WARMUP_THRESHOLD': '-10.0',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_cp_s32': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_CP': '1',
        'FLASHQLA_CP_MAX_LOCAL_CHUNKS': '32',
        'FLASHQLA_CP_MIN_CHUNKS': '512',
        'FLASHQLA_CP_WARMUP_THRESHOLD': '-10.0',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_cp_s64': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_CP': '1',
        'FLASHQLA_CP_MAX_LOCAL_CHUNKS': '64',
        'FLASHQLA_CP_MIN_CHUNKS': '512',
        'FLASHQLA_CP_WARMUP_THRESHOLD': '-10.0',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_cp_torch': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_CP': '1',
        'FLASHQLA_CP_MAX_LOCAL_CHUNKS': '32',
        'FLASHQLA_CP_MIN_CHUNKS': '512',
        'FLASHQLA_CP_WARMUP_THRESHOLD': '-10.0',
        'FLASHQLA_CP_CORRECT_H0_TORCH': '1',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_cp_exact': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_CP_EXACT': '1',
        'FLASHQLA_CP_EXACT': '1',
        'FLASHQLA_CP_MAX_LOCAL_CHUNKS': '16',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'qwen397_native_512': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '512',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h',
    },
    'qwen397_native_512_noh': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '512',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load',
    },
    'ag_256_lh': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h',
    },
    'ag_256_lho': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h,o',
    },
    'ag_128_lh': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '64',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '128',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h',
    },
    'ag_b128_256_lh': {
        'FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE': '1',
        'FLASHQLA_BLACKWELL_NATIVE_KERNELS': 'fwd,kkt',
        'FLASHQLA_BLACKWELL_FWD_POLICY': 'native',
        'FLASHQLA_BLACKWELL_BLOCK_DV': '128',
        'FLASHQLA_BLACKWELL_FWD_THREADS': '256',
        'FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS': 'load,h',
    },
}

ENV_TO_CLEAR = (
    "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE",
    "FLASHQLA_BLACKWELL_NATIVE",
    "FLASHQLA_BLACKWELL_NATIVE_KERNELS",
    "FLASHQLA_ENABLE_BLACKWELL_FWD_NATIVE_VARLEN",
    "FLASHQLA_BLACKWELL_BLOCK_DV",
    "FLASHQLA_BLACKWELL_MIN_BLOCK_DV",
    "FLASHQLA_BLACKWELL_ALLOW_BLOCK_DV32",
    "FLASHQLA_BLACKWELL_TMEM_WIDTH",
    "FLASHQLA_BLACKWELL_FWD_THREADS",
    "FLASHQLA_BLACKWELL_FWD_SYNC_BARRIERS",
    "FLASHQLA_BLACKWELL_FWD_POLICY",
    "FLASHQLA_AUTOCP",
    "FLASHQLA_BLACKWELL_CP",
    "FLASHQLA_BLACKWELL_CP_EXACT",
    "FLASHQLA_CP_EXACT",
    "FLASHQLA_CP_WARMUP_THRESHOLD",
    "FLASHQLA_CP_CORRECT_H0_TORCH",
    "FLASHQLA_CP_MAX_LOCAL_CHUNKS",
    "FLASHQLA_CP_MIN_CHUNKS",
    "FLASHQLA_BLACKWELL_FWD_EXPERIMENT",
    "FLASHQLA_BLACKWELL_FWD_MAX_ITERS",
    "FLASHQLA_BLACKWELL_KKT_EXPERIMENT",
    "FLASHQLA_BLACKWELL_PRETRANSFORM_A",
    "FLASHQLA_CORRECTNESS_REPEATS",
)


SHAPE_RE = re.compile(r"Shape: B=(?P<b>\d+) Hk=(?P<hk>\d+) Hv=(?P<hv>\d+) T=(?P<t>\d+)")
TOTAL_RE = re.compile(r"^total\s+(?P<fla>[0-9.]+|NaN)\s+(?P<qla>[0-9.]+|NaN)\s*$")
SPEEDUP_RE = re.compile(r"Speed up:\s+(?P<speedup>[0-9.]+)x")
KERNEL_RE = re.compile(r"'(?P<kernel>tilelang_[^']+_kernel)'")
PROFILE_RE = re.compile(
    r"^\[fwd\]\s+(?P<name>csum|solve|wu|gdr|o|cp-w|cp-h|cp-c)\s+"
    r"(?P<fla>[0-9.]+|NaN)\s+(?P<qla>[0-9.]+|NaN)\s*$"
)


@dataclass
class RunResult:
    policy: str
    shape: str
    nkh: int
    nvh: int
    batch_size: int | None
    t: int | None
    fla_ms: float | None
    qla_ms: float | None
    speedup: float | None
    fla_csum_ms: float | None
    qla_csum_ms: float | None
    fla_solve_ms: float | None
    qla_solve_ms: float | None
    fla_wu_ms: float | None
    qla_wu_ms: float | None
    fla_gdr_ms: float | None
    qla_gdr_ms: float | None
    fla_o_ms: float | None
    qla_o_ms: float | None
    fla_cp_w_ms: float | None
    qla_cp_w_ms: float | None
    fla_cp_h_ms: float | None
    qla_cp_h_ms: float | None
    fla_cp_c_ms: float | None
    qla_cp_c_ms: float | None
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
    current_b: int | None = None
    current_t: int | None = None
    pending_total: tuple[float | None, float | None] | None = None
    pending_profile: dict[str, tuple[float | None, float | None]] = {}

    for line in text.splitlines():
        shape_match = SHAPE_RE.search(line)
        if shape_match:
            current_b = int(shape_match.group("b"))
            current_t = int(shape_match.group("t"))
            pending_total = None
            pending_profile = {}
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

        profile_match = PROFILE_RE.match(line.strip())
        if profile_match:
            name = profile_match.group("name").replace("-", "_")
            pending_profile[name] = (
                _parse_float(profile_match.group("fla")),
                _parse_float(profile_match.group("qla")),
            )
            continue

        speedup_match = SPEEDUP_RE.search(line)
        if speedup_match and current_t is not None and pending_total is not None:
            row = {
                "t": current_t,
                "batch_size": current_b,
                "fla_ms": pending_total[0],
                "qla_ms": pending_total[1],
                "speedup": float(speedup_match.group("speedup")),
            }
            for name in ("csum", "solve", "wu", "gdr", "o", "cp_w", "cp_h", "cp_c"):
                values = pending_profile.get(name, (None, None))
                row[f"fla_{name}_ms"] = values[0]
                row[f"qla_{name}_ms"] = values[1]
            rows.append(row)
            pending_total = None

    return rows, sorted(kernels)


def _error_tail(text: str, max_lines: int = 24) -> str:
    interesting = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and (
            line.startswith("Shape:")
            or line.lstrip().startswith("File ")
            or "fwd correctness repeat:" in line
            or line.startswith("s_qla:")
            or line.startswith("o_qla:")
            or "max_idx=" in line
            or "top_heads=" in line
            or "top_chunks=" in line
            or "Traceback" in line
            or "Error" in line
            or "ERROR" in line
            or "Exception" in line
            or "AssertionError" in line
            or "RuntimeError" in line
            or "ValueError" in line
            or "AttributeError" in line
            or "OutOfMemoryError" in line
            or "timeout" in line.lower()
        )
    ]
    tail = interesting[-max_lines:]
    return " | ".join(tail)


def _make_filtered_settings(
    base_set: str,
    seqlens: list[int] | None,
    batch_sizes: list[int] | None,
    log_dir: Path,
) -> tuple[str, Path]:
    source = Path("tests/settings") / f"{base_set}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing settings file: {source}")

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if seqlens is not None and int(row["num_tokens"]) not in seqlens:
                continue
            if batch_sizes is None:
                rows.append(dict(row))
            else:
                for batch_size in batch_sizes:
                    new_row = dict(row)
                    new_row["batch_size"] = str(batch_size)
                    rows.append(new_row)
        fieldnames = reader.fieldnames

    if not rows:
        raise ValueError(
            f"No rows in {source} match --seqlens={seqlens} "
            f"--batch-sizes={batch_sizes}"
        )
    if fieldnames is None:
        raise ValueError(f"Settings file has no header: {source}")

    suffix_parts = [str(os.getpid())]
    if seqlens is not None:
        suffix_parts.append("t" + "_".join(map(str, seqlens)))
    if batch_sizes is not None:
        suffix_parts.append("b" + "_".join(map(str, batch_sizes)))
    name = "_blackwell_policy_" + "_".join(suffix_parts)
    target = Path("tests/settings") / f"{name}.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (log_dir / f"{name}.csv").write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    return name, target


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
    force_cp = (
        env.get("FLASHQLA_BLACKWELL_CP") == "1"
        or env.get("FLASHQLA_BLACKWELL_CP_EXACT") == "1"
    )
    if args.no_cp and not force_cp:
        cmd.append("--no-cp")
    if args.no_h0:
        cmd.append("--no-h0")
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
                batch_size=None,
                t=None,
                fla_ms=None,
                qla_ms=None,
                speedup=None,
                fla_csum_ms=None,
                qla_csum_ms=None,
                fla_solve_ms=None,
                qla_solve_ms=None,
                fla_wu_ms=None,
                qla_wu_ms=None,
                fla_gdr_ms=None,
                qla_gdr_ms=None,
                fla_o_ms=None,
                qla_o_ms=None,
                fla_cp_w_ms=None,
                qla_cp_w_ms=None,
                fla_cp_h_ms=None,
                qla_cp_h_ms=None,
                fla_cp_c_ms=None,
                qla_cp_c_ms=None,
                status=status,
                returncode=returncode,
                error_tail=error_tail,
                log_path=str(log_path),
                kernels=kernel_text,
            )
        ]

    results = [
        RunResult(
            policy=policy,
            shape=shape,
            nkh=nkh,
            nvh=nvh,
            batch_size=row["batch_size"],
            t=int(row["t"]),
            fla_ms=row["fla_ms"],
            qla_ms=row["qla_ms"],
            speedup=row["speedup"],
            fla_csum_ms=row["fla_csum_ms"],
            qla_csum_ms=row["qla_csum_ms"],
            fla_solve_ms=row["fla_solve_ms"],
            qla_solve_ms=row["qla_solve_ms"],
            fla_wu_ms=row["fla_wu_ms"],
            qla_wu_ms=row["qla_wu_ms"],
            fla_gdr_ms=row["fla_gdr_ms"],
            qla_gdr_ms=row["qla_gdr_ms"],
            fla_o_ms=row["fla_o_ms"],
            qla_o_ms=row["qla_o_ms"],
            fla_cp_w_ms=row["fla_cp_w_ms"],
            qla_cp_w_ms=row["qla_cp_w_ms"],
            fla_cp_h_ms=row["fla_cp_h_ms"],
            qla_cp_h_ms=row["qla_cp_h_ms"],
            fla_cp_c_ms=row["fla_cp_c_ms"],
            qla_cp_c_ms=row["qla_cp_c_ms"],
            status=status,
            returncode=returncode,
            error_tail=error_tail,
            log_path=str(log_path),
            kernels=kernel_text,
        )
        for row in parsed
    ]
    if status != "ok" and len(results) > 1:
        for row in results[:-1]:
            row.status = "ok"
            row.returncode = 0
            row.error_tail = ""
    return results


def _write_csv(path: Path, rows: list[RunResult]) -> None:
    fieldnames = (
        "policy",
        "shape",
        "nkh",
        "nvh",
        "batch_size",
        "t",
        "fla_ms",
        "qla_ms",
        "speedup",
        "fla_csum_ms",
        "qla_csum_ms",
        "fla_solve_ms",
        "qla_solve_ms",
        "fla_wu_ms",
        "qla_wu_ms",
        "fla_gdr_ms",
        "qla_gdr_ms",
        "fla_o_ms",
        "qla_o_ms",
        "fla_cp_w_ms",
        "qla_cp_w_ms",
        "fla_cp_h_ms",
        "qla_cp_h_ms",
        "fla_cp_c_ms",
        "qla_cp_c_ms",
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
        speedups = [
            row.speedup
            for row in group
            if row.speedup is not None and row.status == "ok"
        ]
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

    failed = [row for row in rows if row.status != "ok"]
    if failed:
        print("\nFailures", flush=True)
        for row in failed:
            batch = f" b={row.batch_size}" if row.batch_size is not None else ""
            print(
                f"{row.policy:16s} {row.shape:6s} "
                f"t={row.t if row.t is not None else 'unknown'}{batch} "
                f"status={row.status} rc={row.returncode} "
                f"tail={row.error_tail}",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies",
        default="qwen397_native",
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
    parser.add_argument(
        "--seqlens",
        default="",
        help=(
            "Optional comma-separated num_tokens filter for the selected settings "
            "file, e.g. 4096 or 4096,8192. A temporary filtered settings file is "
            "created under tests/settings and copied to the log directory."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        default="",
        help=(
            "Optional comma-separated batch sizes. Matching settings rows are "
            "duplicated with these batch_size values."
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--correctness-repeats", type=int, default=1000)
    parser.add_argument("--out", default="blackwell_policy.csv")
    parser.add_argument("--log-dir", default="blackwell_policy_logs")
    parser.add_argument("--no-cp", action="store_true", default=True)
    parser.add_argument("--with-cp", action="store_false", dest="no_cp")
    parser.add_argument(
        "--no-h0",
        action="store_true",
        help="Pass --no-h0 to tests/test_gdr.py to isolate initial-state issues.",
    )
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
    if not args.no_cp:
        native_fwd_cp = [
            policy
            for policy in selected_policies
            if "fwd" in POLICIES[policy].get("FLASHQLA_BLACKWELL_NATIVE_KERNELS", "")
            and POLICIES[policy].get("FLASHQLA_BLACKWELL_CP") != "1"
            and POLICIES[policy].get("FLASHQLA_BLACKWELL_CP_EXACT") != "1"
        ]
        if native_fwd_cp:
            print(
                "[warn] --with-cp is unsupported for Blackwell native fwd policies "
                "now that Hopper fallback is disabled: "
                + ",".join(native_fwd_cp),
                flush=True,
            )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    filtered_settings_path: Path | None = None
    seqlens = [int(item) for item in _split_csv(args.seqlens)] if args.seqlens else None
    batch_sizes = (
        [int(item) for item in _split_csv(args.batch_sizes)]
        if args.batch_sizes
        else None
    )
    if seqlens is not None or batch_sizes is not None:
        args.set, filtered_settings_path = _make_filtered_settings(
            args.set, seqlens, batch_sizes, log_dir
        )

    try:
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
    finally:
        if filtered_settings_path is not None:
            filtered_settings_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
