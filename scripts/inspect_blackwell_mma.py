#!/usr/bin/env python3
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Inspect FlashQLA generated code on Blackwell.

This script is intended to be run on the target B200/B300 machine. It triggers
one FlashQLA forward compile, scans recently generated CUDA artifacts, and
reports whether their SASS/text contains Blackwell tcgen05 instructions or
Hopper WGMMA instructions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


TCGEN_PATTERNS = ("tcgen05", "tcgen")
WGMMA_PATTERNS = ("wgmma", "wgmma.mma_async")
HMMA_PATTERNS = ("hmma", "hmma.16816", "hmma.1688")
TMEM_PATTERNS = ("tmem", "tcgen05.alloc", "tcgen05.commit")
ARTIFACT_SUFFIXES = (".cubin", ".so", ".ptx", ".sass", ".cu", ".ll")
UNRELATED_PATH_PARTS = (
    "flashinfer",
    "gdrcopy",
    "gdn_prefill",
)


def _run(cmd: list[str], timeout: int = 120) -> str:
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _artifact_roots(extra_roots: list[str]) -> list[Path]:
    candidates = [
        os.environ.get("TILELANG_CACHE_DIR"),
        os.environ.get("TL_CACHE_DIR"),
        os.environ.get("CUDA_CACHE_PATH"),
        "~/.cache",
        "/tmp",
        "/tmp/tvm-debug-mode-tempdirs",
        "/var/tmp",
        str(Path.cwd()),
    ]
    candidates.extend(extra_roots)
    roots = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path not in roots:
            roots.append(path)
    return roots


def _iter_artifacts(roots: list[Path], since: float | None) -> list[Path]:
    artifacts = []
    for root in roots:
        try:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in ARTIFACT_SUFFIXES:
                    continue
                if since is not None and path.stat().st_mtime < since:
                    continue
                artifacts.append(path)
        except OSError:
            continue
    return sorted(set(artifacts), key=lambda p: p.stat().st_mtime, reverse=True)


def _latest_tvm_debug_dir() -> Path | None:
    root = Path("/tmp/tvm-debug-mode-tempdirs")
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.stat().st_mtime)


def _is_relevant_artifact(path: Path) -> bool:
    lower = str(path).lower()
    if any(part in lower for part in UNRELATED_PATH_PARTS):
        return False
    if "tvm-debug-mode-tempdirs" in lower:
        return path.name in {
            "tvm_kernels.cubin",
            "tvm_kernels.ptx",
            "tvm_kernels.cu",
            "tvm_kernels.sass",
            "tvm_kernels.ll",
        }
    if "tilelang" in lower or "flash_qla" in lower or "flashqla" in lower:
        return True
    return False


def _artifact_text(path: Path) -> str:
    if path.suffix == ".cubin" and shutil.which("nvdisasm"):
        return _run(["nvdisasm", str(path)])
    if path.suffix == ".so" and shutil.which("cuobjdump"):
        return _run(["cuobjdump", "--dump-sass", str(path)])
    if path.suffix in {".ptx", ".sass"}:
        try:
            return path.read_text(errors="ignore")
        except OSError:
            return ""
    # Fallback: `strings` can still find mnemonic text in some artifacts.
    if shutil.which("strings"):
        return _run(["strings", str(path)])
    return ""


def _classify(text: str) -> tuple[bool, bool, bool, bool]:
    lower = text.lower()
    has_tcgen = any(pattern in lower for pattern in TCGEN_PATTERNS)
    has_wgmma = any(pattern in lower for pattern in WGMMA_PATTERNS)
    has_hmma = any(pattern in lower for pattern in HMMA_PATTERNS)
    has_tmem = any(pattern in lower for pattern in TMEM_PATTERNS)
    return has_tcgen, has_wgmma, has_hmma, has_tmem


def _trigger_flashqla_compile(args: argparse.Namespace) -> None:
    import torch

    from flash_qla import chunk_gated_delta_rule_fwd
    from flash_qla.utils import l2norm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this on the B200/B300 host.")

    device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    print(f"device={torch.cuda.get_device_name(device_index)} sm_{major}{minor}")

    q = l2norm(
        torch.randn(
            (args.batch, args.tokens, args.q_heads, args.head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
    )
    k = l2norm(torch.randn_like(q))
    v = torch.randn(
        (args.batch, args.tokens, args.v_heads, args.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    g = torch.nn.functional.logsigmoid(
        torch.randn(
            (args.batch, args.tokens, args.v_heads),
            device="cuda",
            dtype=torch.float32,
        )
    )
    beta = torch.randn_like(g)
    h0 = torch.randn(
        (args.batch, args.v_heads, args.head_dim, args.head_dim),
        device="cuda",
        dtype=torch.float32,
    )

    torch.cuda.synchronize()
    chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=args.head_dim ** -0.5,
        initial_state=h0,
        output_final_state=True,
        output_h=False,
        auto_cp=args.auto_cp,
    )
    torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-run", action="store_true", help="Only inspect artifacts")
    parser.add_argument("--root", action="append", default=[], help="Extra scan root")
    parser.add_argument("--all", action="store_true", help="Scan all artifacts")
    parser.add_argument(
        "--latest-tvm-dir",
        action="store_true",
        help="Inspect only the newest /tmp/tvm-debug-mode-tempdirs run directory",
    )
    parser.add_argument(
        "--since-minutes",
        type=float,
        default=None,
        help="Inspect artifacts modified within the last N minutes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of matching artifacts to print",
    )
    parser.add_argument(
        "--include-unrelated",
        action="store_true",
        help="Also classify unrelated artifacts such as FlashInfer/GDRCopy",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--q-heads", type=int, default=2)
    parser.add_argument("--v-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--auto-cp", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    if not args.no_run:
        _trigger_flashqla_compile(args)

    latest_tvm_dir = _latest_tvm_debug_dir() if args.latest_tvm_dir else None
    roots = [latest_tvm_dir] if latest_tvm_dir is not None else _artifact_roots(args.root)
    if args.since_minutes is not None:
        since = (datetime.now() - timedelta(minutes=args.since_minutes)).timestamp()
    else:
        since = None if args.all or args.no_run or args.latest_tvm_dir else start_time
    artifacts = _iter_artifacts(roots, since)
    if not artifacts and since is not None:
        print("no recent artifacts found; falling back to full artifact scan")
        artifacts = _iter_artifacts(roots, None)
    if not args.include_unrelated:
        artifacts = [path for path in artifacts if _is_relevant_artifact(path)]
    print(f"scanned_roots={':'.join(str(root) for root in roots)}")
    if latest_tvm_dir is not None:
        print(f"latest_tvm_dir={latest_tvm_dir}")
    print(f"candidate_artifacts={len(artifacts)}")

    hits = []
    for path in artifacts:
        text = _artifact_text(path)
        has_tcgen, has_wgmma, has_hmma, has_tmem = _classify(text)
        if has_tcgen or has_wgmma or has_hmma or has_tmem:
            hits.append((path, has_tcgen, has_wgmma, has_hmma, has_tmem))

    for path, has_tcgen, has_wgmma, has_hmma, has_tmem in hits[: args.limit]:
        flags = []
        if has_tcgen:
            flags.append("tcgen05/tcgen")
        if has_wgmma:
            flags.append("wgmma")
        if has_hmma:
            flags.append("hmma")
        if has_tmem:
            flags.append("tmem")
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        print(f"HIT {'+'.join(flags)} mtime={mtime} {path}")
    if len(hits) > args.limit:
        print(f"... omitted {len(hits) - args.limit} hits; increase --limit to show more")

    if any(has_tcgen for _, has_tcgen, _, _, _ in hits):
        print("RESULT: Blackwell tensor core instructions detected.")
        return 0
    if any(has_wgmma for _, _, has_wgmma, _, _ in hits):
        print("RESULT: Hopper WGMMA instructions detected; Blackwell-native path is missing.")
        return 2
    if any(has_hmma for _, _, _, has_hmma, _ in hits):
        print(
            "RESULT: legacy HMMA instructions detected; Blackwell-native "
            "tcgen05/TMEM path is missing."
        )
        return 3

    print(
        "RESULT: inconclusive. Ensure cuobjdump/nvdisasm is installed and rerun with "
        "--all or --root pointing at the TileLang/TVM cache directory. If TVM "
        "debug artifacts are created and removed too quickly, run the benchmark "
        "once, then rerun this script with --no-run --all."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
