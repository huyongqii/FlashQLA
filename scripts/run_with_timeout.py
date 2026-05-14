#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Run a command and kill its process group on timeout."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.cmd:
        raise ValueError("missing command")

    cmd = args.cmd
    if cmd[0] == "--":
        cmd = cmd[1:]
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"timeout after {args.timeout}s; killing process group", flush=True)
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
