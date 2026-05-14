#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Run B200 FlashQLA GDR variants in isolated subprocesses."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


VARIANTS = {
    "compat": {},
    "kkt": {
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "kkt",
    },
    "kkt_fwd": {
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "kkt,fwd",
    },
    "fwd": {
        "FLASHQLA_BLACKWELL_NATIVE": "1",
        "FLASHQLA_BLACKWELL_NATIVE_KERNELS": "fwd",
    },
}


def _run_variant(name: str, args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env.pop("FLASHQLA_BLACKWELL_NATIVE", None)
    env.pop("FLASHQLA_BLACKWELL_NATIVE_KERNELS", None)
    env.update(VARIANTS[name])

    cmd = [
        sys.executable,
        "tests/test_gdr.py",
        "--set",
        args.set,
        "--skip-bwd",
    ]
    if args.hide_acc:
        cmd.append("--hide-acc")
    if args.hide_lat:
        cmd.append("--hide-lat")
    if args.seqlen is not None:
        cmd.extend(["--seqlen", str(args.seqlen)])
    if args.nkh is not None:
        cmd.extend(["--nkh", str(args.nkh)])
    if args.nvh is not None:
        cmd.extend(["--nvh", str(args.nvh)])

    print("=" * 80, flush=True)
    print(f"variant={name} env={VARIANTS[name]}", flush=True)
    print("cmd=" + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    print(f"variant={name} returncode={proc.returncode}", flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="compat,fwd,kkt",
        help="Comma-separated variants: compat,fwd,kkt,kkt_fwd",
    )
    parser.add_argument("--set", default="profile")
    parser.add_argument("--seqlen", type=int, default=None)
    parser.add_argument("--nkh", type=int, default=None)
    parser.add_argument("--nvh", type=int, default=None)
    parser.add_argument("--hide-acc", action="store_true")
    parser.add_argument("--hide-lat", action="store_true")
    args = parser.parse_args()

    selected = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in selected if item not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid: {sorted(VARIANTS)}")

    rc = 0
    for name in selected:
        rc = max(rc, _run_variant(name, args))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
