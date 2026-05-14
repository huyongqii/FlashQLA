# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

import os
import subprocess
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))

rev = os.getenv("QLA_VERSION_SUFFIX", "")
if not rev:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
    except Exception:
        rev = ""

setup(
    name="flash_qla",
    version="0.1.0" + rev,
    description="FlashQLA: Fused TileLang kernels for Linear Attention",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.8",
        # tilelang 0.1.9 is the latest version available on internal PyPI and
        # is verified to identify sm_100 (B200) correctly. Older 0.1.8 was
        # used originally on H200; we relax the bound so a single install
        # covers both Hopper (sm_90) and Blackwell (sm_100) archs.
        "tilelang>=0.1.9",
        "apache-tvm-ffi>=0.1.9",
    ],
    zip_safe=False,
)
