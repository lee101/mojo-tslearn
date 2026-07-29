"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "capi.mojo")
LIB = os.environ.get("MOJOTSLEARN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-tslearn.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mts_dtw": ([I, I, I, I, I, I, I, I], F),
    "mts_dtw_backtrack": ([I, I, I, I, I], I),
    "mts_soft_dtw": ([I, I, I, I, I, F, I], F),
    "mts_dtw_from_cost": ([I, I, I, I, I, I], F),
    "mts_cdist_dtw": ([I, I, I, I, I, I, I, I, I, I], None),
    "mts_cdist_soft_dtw": ([I, I, I, I, I, I, I, F, I, I], None),
    "mts_cdist_soft_dtw_gpu": ([I, I, I, I, I, I, I, F, I], I),
    "mts_cdist_euclidean": ([I, I, I, I, I, I], None),
    "mts_dba_accumulate": ([I, I, I, I, I, F, I, I, I], F),
}


class BuildError(RuntimeError):
    pass


def _build() -> str:
    if os.environ.get("MOJOTSLEARN_LIB"):
        if not os.path.exists(LIB):
            raise BuildError(f"MOJOTSLEARN_LIB does not exist: {LIB}")
        return LIB
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    pixi = shutil.which("pixi")
    if pixi:
        cmd = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    else:
        cmd = ["bash", os.path.join(ROOT, "build", "build.sh")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(_build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIBRARY, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIBRARY


def f64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def main() -> int:
    print(_build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
