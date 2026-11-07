"""Preload CUDA .so files so jaxlib's cuSPARSE version check can dlopen them.

Installed into the vadar venv as sitecustomize.py. LD_LIBRARY_PATH alone is not
enough: jaxlib looks up libcusparse by soname before that path is always used.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path

_SONAMES = (
    "libnvJitLink.so.12",
    "libcudart.so.12",
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcusparse.so.12",
    "libcusparse.so",
    "libcudnn.so.9",
    "libcudnn.so.8",
)


def _search_roots() -> list[Path]:
    roots = [Path(sys.prefix)]
    for extra in ("/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
        p = Path(extra)
        if p.is_dir():
            roots.append(p)
    return roots


def _lib_dirs() -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()
    for root in _search_roots():
        pattern = str(root / "**" / "libcusparse.so*")
        for so in glob.glob(pattern, recursive=True):
            d = str(Path(so).parent)
            if d not in seen:
                seen.add(d)
                dirs.append(d)
        for d in glob.glob(str(root / "**" / "nvidia" / "*" / "lib"), recursive=True):
            if os.path.isdir(d) and d not in seen:
                seen.add(d)
                dirs.append(d)
    return dirs


def _preload() -> None:
    dirs = _lib_dirs()
    if dirs:
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + ([cur] if cur else []))
    for d in dirs:
        for name in _SONAMES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


_preload()
