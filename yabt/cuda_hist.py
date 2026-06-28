"""Custom CUDA histogram kernel for the level-wise GPU grower.

The level-wise grower's hot GPU op is the per-level histogram build. The pure
torch version (``grow_tree_levelwise.scatter_hist``) materializes an (n*F) int64
index tensor and expanded value tensors and runs three ``scatter_add_`` passes;
that is bandwidth-heavy and launch-heavy. ``_cuda/hist.cu`` does the same atomic
accumulation in a single fused kernel with no big temporaries.

Compiled once per process with :func:`torch.utils.cpp_extension.load` (cached
under ``TORCH_EXTENSIONS_DIR``), mirroring the lazy compile-on-first-use approach
of the C grower. Any failure to build/load leaves :func:`is_available` False and
the caller falls back to the torch scatter -- a perf fallback, not a correctness
loss.
"""

from __future__ import annotations

import os
import sys
import threading

import torch


def _ensure_ninja_on_path() -> None:
    """torch's cpp_extension shells out to the ``ninja`` executable. The pip/uv
    ``ninja`` package installs it next to the interpreter, but that dir is only on
    PATH when the venv is *activated* -- running ``.venv/bin/python`` directly
    leaves it off. Prepend the interpreter's bin dir so the build finds ninja
    regardless (a no-op if it is already there or ninja isn't installed)."""
    bindir = os.path.dirname(os.path.abspath(sys.executable))
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bindir and bindir not in parts:
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_cuda", "hist.cu")
_LOCK = threading.Lock()
_MOD = None  # cached extension module, or False once a build attempt has failed


def _load():
    global _MOD
    if _MOD is not None:
        if _MOD is False:
            raise RuntimeError("CUDA histogram kernel unavailable (build failed)")
        return _MOD
    with _LOCK:
        if _MOD is not None:
            if _MOD is False:
                raise RuntimeError("CUDA histogram kernel unavailable (build failed)")
            return _MOD
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device")
            _ensure_ninja_on_path()
            from torch.utils.cpp_extension import load
            _MOD = load(
                name="yabt_hist_cuda",
                sources=[_SRC],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
        except Exception:
            _MOD = False
            raise
        return _MOD


def is_available() -> bool:
    """True if the CUDA histogram kernel can be compiled and loaded."""
    try:
        _load()
        return True
    except Exception:
        return False


def build_hist(binned: torch.Tensor, slot: torch.Tensor, g: torch.Tensor,
               h: torch.Tensor, w: torch.Tensor, K: int, F: int, B: int) -> torch.Tensor:
    """Fused (3, K, F, B) histogram: row ``i`` adds ``(g[i], h[i], w[i])`` into
    bin ``binned[i, f]`` of node slot ``slot[i]`` for every feature ``f``.

    ``binned`` is (n, F) uint8 row-major; ``slot`` int64 (n,); ``g/h/w`` float32
    (n,). Equivalent to the torch ``scatter_hist`` (same atomic accumulation), so
    results match up to float atomic-add ordering -- which torch's CUDA
    ``scatter_add_`` is also subject to."""
    mod = _load()
    return mod.hist_build(binned, slot, g, h, w, int(K), int(F), int(B))
