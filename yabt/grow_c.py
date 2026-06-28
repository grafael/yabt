"""OpenMP-parallel C grower: a multi-core drop-in for :func:`grow_tree_numba`.

The Numba grower (:mod:`yabt.grow_numba`) is a single thread of tight scalar
loops; on a many-core box it leaves most of the machine idle while XGBoost and
LightGBM saturate every core. ``_cgrow/grow.c`` reimplements the same leaf-wise
grower with the two hot loops -- the dense histogram build and the per-feature
split search -- parallelized across features with OpenMP. The split math is
bit-identical (per-feature accumulation order is preserved), so trees match the
Numba/torch grower; the only difference is wall-clock on multi-core CPUs.

The C source is compiled once to a cached shared library next to this module
(keyed by a hash of the source + the exact build command) and called through
ctypes, mirroring the project's compile-on-the-dev-box approach for Numba. The
builder tries a list of platform-appropriate compiler/flag candidates and uses
the first that compiles and loads:

  * Linux / other Unix: GCC or Clang with ``-fopenmp`` (libgomp), ``.so``.
  * macOS: Apple Clang with ``-Xpreprocessor -fopenmp`` against Homebrew
    ``libomp`` (auto-located via ``brew --prefix`` or the usual install dirs),
    or a Homebrew GCC/Clang with plain ``-fopenmp``; ``.so`` (a Mach-O dylib,
    loaded by full path).
  * Windows: MinGW/Clang ``-fopenmp`` or MSVC ``cl /openmp``, ``.dll``.

If none work (no compiler, no OpenMP runtime), :func:`is_available` returns
False and the caller falls back to the single-threaded Numba grower. Falling
back is not a correctness loss -- single-threaded C is at parity with Numba; the
C library's only edge is multi-core scaling.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np
import torch

from .binning import MAX_BINS
from .tree import Tree, TreeParams, LEAF

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_cgrow", "grow.c")
_LOCK = threading.Lock()
_LIB = None  # cached ctypes handle (or False once all build attempts have failed)


def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _compilers():
    """Candidate C compilers: ``$CC`` first (if set), then platform defaults,
    keeping only those actually found on PATH."""
    cands = [os.environ.get("CC")]
    if sys.platform == "darwin":
        cands += ["cc", "clang", "gcc"]
    elif sys.platform.startswith("win"):
        cands += ["gcc", "clang", "cc"]
    else:
        cands += ["cc", "gcc", "clang"]
    return [c for c in _dedup(cands) if shutil.which(c)]


def _libomp_prefixes():
    """Homebrew ``libomp`` install prefixes on macOS (most specific first).

    Apple's Clang compiles OpenMP via ``-Xpreprocessor -fopenmp`` but ships no
    OpenMP runtime; ``brew install libomp`` provides it. We locate it through
    ``brew --prefix`` and the standard Apple-Silicon / Intel install dirs."""
    prefixes = []
    brew = shutil.which("brew")
    if brew:
        try:
            r = subprocess.run([brew, "--prefix", "libomp"],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                prefixes.append(r.stdout.strip())
        except Exception:
            pass
    prefixes += ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
    return _dedup([p for p in prefixes if os.path.isdir(p)])


def _gnu_argv(cc, omp, arch, libs):
    """GCC/Clang-style command (Linux, macOS, MinGW)."""
    def fn(src, out):
        return [cc, "-O3", "-shared", "-fPIC", "-std=c11",
                *arch, *omp, src, "-o", out, "-lm", *libs]
    return fn


def _msvc_argv(cc):
    """MSVC ``cl`` command: ``/LD`` -> DLL, ``/openmp`` -> vcomp runtime. Object
    files land in the (temp) working dir; ``/Fe:`` names the DLL output."""
    def fn(src, out):
        return [cc, "/nologo", "/O2", "/openmp", "/std:c11", "/LD", src,
                "/Fe:" + out]
    return fn


def _candidates():
    """Ordered (name, ext, argv_fn) build candidates for this platform."""
    out = []
    if sys.platform == "darwin":
        prefixes = _libomp_prefixes()
        for cc in _compilers():
            # Apple Clang: OpenMP via -Xpreprocessor + an explicit libomp.
            for prefix in prefixes:
                omp = ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"]
                libs = [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"]
                for arch in (["-mcpu=native"], ["-march=native"], []):
                    tag = f"{cc}+libomp {' '.join(arch) or 'noarch'}"
                    out.append((tag, ".so", _gnu_argv(cc, omp, arch, libs)))
            # Homebrew GCC/Clang may accept -fopenmp directly (bundled runtime).
            for arch in (["-mcpu=native"], ["-march=native"], []):
                tag = f"{cc} {' '.join(arch) or 'noarch'}"
                out.append((tag, ".so", _gnu_argv(cc, ["-fopenmp"], arch, [])))
    elif sys.platform.startswith("win"):
        for cc in _compilers():  # MinGW / clang
            for arch in (["-march=native"], []):
                tag = f"{cc} {' '.join(arch) or 'noarch'}"
                out.append((tag, ".dll", _gnu_argv(cc, ["-fopenmp"], arch, [])))
        if shutil.which("cl"):  # MSVC (needs a developer command prompt env)
            out.append(("cl /openmp", ".dll", _msvc_argv("cl")))
    else:  # linux and other unix
        for cc in _compilers():
            for arch in (["-march=native"], []):
                tag = f"{cc} {' '.join(arch) or 'noarch'}"
                out.append((tag, ".so", _gnu_argv(cc, ["-fopenmp"], arch, [])))
    return out


def _artifact_path(ext: str, argv_fn) -> str:
    """Cached library path, keyed by the source + the exact build command so a
    different compiler/flag set (or edited source) gets its own artifact."""
    with open(_SRC, "rb") as fh:
        src = fh.read()
    sig = " ".join(argv_fn("SRC", "OUT")).encode()
    key = hashlib.sha1(src + sig).hexdigest()[:16]
    return os.path.join(_HERE, "_cgrow", f"grow_{key}{ext}")


def _configure(lib) -> None:
    """Declare ctypes signatures for the loaded library."""
    lib.capply.restype = None
    lib.capply.argtypes = [
        ctypes.c_void_p,  # X (f32, n*F row-major)
        ctypes.c_void_p,  # feature (i64)
        ctypes.c_void_p,  # threshold (f32)
        ctypes.c_void_p,  # left (i64)
        ctypes.c_void_p,  # right (i64)
        ctypes.c_int64,   # n
        ctypes.c_int,     # F
        ctypes.c_int,     # n_threads
        ctypes.c_void_p,  # out_node (i64)
    ]
    lib.cgrow.restype = ctypes.c_int
    lib.cgrow.argtypes = [
        ctypes.c_void_p,  # binned (uint8)
        ctypes.c_void_p,  # grad (f32)
        ctypes.c_void_p,  # hess (f32)
        ctypes.c_void_p,  # fmask (uint8)
        ctypes.c_void_p,  # imat (f32)
        ctypes.c_float,   # ib
        ctypes.c_int,     # use_imat
        ctypes.c_void_p,  # indptr (i64)
        ctypes.c_void_p,  # indices (i32)
        ctypes.c_void_p,  # data (i32)
        ctypes.c_void_p,  # default_bin (i32)
        ctypes.c_int,     # use_sparse
        ctypes.c_void_p,  # nbins (i32)
        ctypes.c_int64,   # n
        ctypes.c_int,     # F
        ctypes.c_int,     # B
        ctypes.c_float,   # lam
        ctypes.c_float,   # gamma
        ctypes.c_float,   # mcw
        ctypes.c_int,     # msl
        ctypes.c_float,   # lr
        ctypes.c_int,     # max_leaves
        ctypes.c_int,     # max_depth
        ctypes.c_int,     # n_threads
        ctypes.c_void_p,  # out_feature (i64)
        ctypes.c_void_p,  # out_thr_bin (i64)
        ctypes.c_void_p,  # out_left (i64)
        ctypes.c_void_p,  # out_right (i64)
        ctypes.c_void_p,  # out_value (f32)
        ctypes.c_void_p,  # out_depth (i64)
    ]


def _load():
    """Compile (if needed) and load the shared library; cache the handle.

    Tries each platform candidate in order and returns the first that compiles
    and loads. Raises (and caches the failure) if none work, so the caller falls
    back to the Numba grower.
    """
    global _LIB
    if _LIB is not None:
        if _LIB is False:
            raise RuntimeError("C grower unavailable (build failed on this machine)")
        return _LIB
    with _LOCK:
        if _LIB is not None:
            if _LIB is False:
                raise RuntimeError("C grower unavailable (build failed on this machine)")
            return _LIB
        errors = []
        for name, ext, argv_fn in _candidates():
            try:
                out = _artifact_path(ext, argv_fn)
                if not os.path.exists(out):
                    with tempfile.TemporaryDirectory() as wd:
                        subprocess.run(argv_fn(_SRC, out), check=True,
                                       capture_output=True, text=True, cwd=wd)
                    if not os.path.exists(out):
                        raise RuntimeError("compiler reported success but produced no library")
                lib = ctypes.CDLL(out)
                _configure(lib)
                _LIB = lib
                return _LIB
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                errors.append(f"[{name}] {type(exc).__name__}: {exc}")
        _LIB = False
        detail = "\n  ".join(errors) if errors else "no C compiler found on PATH"
        raise RuntimeError("C grower unavailable; build candidates failed:\n  " + detail)


def is_available() -> bool:
    """True if the C grower can be compiled and loaded on this machine."""
    try:
        _load()
        return True
    except Exception:
        return False


def prebuild() -> bool:
    """Eagerly compile and load the C grower, printing a diagnostic report.

    This is the deliberate counterpart to the lazy compile-on-first-``fit``
    path: run it in a Dockerfile, CI step, or post-install hook to surface
    compiler/OpenMP problems up front (and to bake the cached ``.so`` into an
    image) instead of paying the build -- or discovering it is impossible --
    during the first training run. Returns True if the library is usable.
    """
    print(f"platform: {sys.platform}")
    print(f"compilers on PATH: {', '.join(_compilers()) or '(none found)'}")
    try:
        _load()
        # Report the cached artifact that actually loaded.
        for _name, ext, argv_fn in _candidates():
            path = _artifact_path(ext, argv_fn)
            if os.path.exists(path):
                print(f"C grower: available -> {path}")
                break
        else:
            print("C grower: available")
        return True
    except Exception as exc:
        print(f"C grower: UNAVAILABLE\n{exc}")
        print("\nThe Numba grower (single-threaded) will be used instead; this "
              "is a performance fallback, not a correctness loss.")
        return False


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.c_void_p)


def _auto_threads() -> int:
    """Default OpenMP thread count for the grower.

    The grower and the (still-serial) neural-leaf/refine phases run sequentially,
    so there is no concurrent oversubscription with torch's pool. We cap at 8:
    in the thread sweep the per-feature histogram/split parallelism stopped
    scaling past ~8 (memory bandwidth + reduction overhead), and going wider
    oversubscribed SMT siblings and regressed. An explicit ``c_grower_threads``
    or a deliberately raised ``OMP_NUM_THREADS`` (>1) overrides this.
    """
    env = os.environ.get("OMP_NUM_THREADS")
    if env is not None:
        try:
            v = int(env)
            if v > 1:  # caller deliberately raised the cap -> honor it
                return v
        except ValueError:
            pass
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, 8))


def apply_c(
    X: torch.Tensor,
    feature: torch.Tensor,
    threshold: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    n_threads: int = 0,
) -> torch.Tensor:
    """Hard-routing leaf index per row of ``X`` via the OpenMP C kernel.

    Drop-in for the axis-only path of :meth:`yabt.tree.Tree.apply`; the caller
    must ensure the tree has no kernel splits. Returns an int64 tensor on
    ``X.device`` (CPU expected -- this is the CPU acceleration path).
    """
    lib = _load()
    if n_threads <= 0:
        n_threads = _auto_threads()
    dev = X.device
    n, F = X.shape
    Xn = np.ascontiguousarray(X.detach().cpu().numpy(), dtype=np.float32)
    feat = np.ascontiguousarray(feature.detach().cpu().numpy(), dtype=np.int64)
    thr = np.ascontiguousarray(threshold.detach().cpu().numpy(), dtype=np.float32)
    lft = np.ascontiguousarray(left.detach().cpu().numpy(), dtype=np.int64)
    rgt = np.ascontiguousarray(right.detach().cpu().numpy(), dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    lib.capply(_ptr(Xn), _ptr(feat), _ptr(thr), _ptr(lft), _ptr(rgt),
               ctypes.c_int64(int(n)), ctypes.c_int(int(F)),
               ctypes.c_int(int(n_threads)), _ptr(out))
    return torch.from_numpy(out).to(dev)


def grow_tree_c(
    binned: torch.Tensor,
    grad: torch.Tensor,
    hess: torch.Tensor,
    binner,
    params: TreeParams,
    feature_mask: torch.Tensor | None = None,
    interaction_matrix: torch.Tensor | None = None,
    interaction_boost: float = 0.5,
    sparse_layout=None,
    n_threads: int = 0,
    binned_fmajor: np.ndarray | None = None,
) -> Tree:
    """Drop-in for the axis path of :func:`yabt.grow_numba.grow_tree_numba`,
    grown by the OpenMP C kernel. ``n_threads<=0`` lets OpenMP pick (env).

    ``binned_fmajor`` optionally supplies the feature-major (F, n) uint8 layout
    of ``binned`` already built; the boosting loop reuses it across rounds (the
    binned matrix is constant when rows are not subsampled), avoiding a transpose
    + copy of the whole matrix every tree."""
    lib = _load()
    if n_threads <= 0:
        n_threads = _auto_threads()
    dev = binned.device
    n, F = binned.shape

    # Feature-major (F, n) layout: the C histogram build keeps each thread's
    # column contiguous (cache/TLB-local), the layout every hist-GBDT lib uses.
    if binned_fmajor is not None:
        bn = binned_fmajor
    else:
        bn = np.ascontiguousarray(binned.detach().cpu().numpy().T, dtype=np.uint8)
    gn = np.ascontiguousarray(grad.detach().cpu().numpy(), dtype=np.float32)
    hn = np.ascontiguousarray(hess.detach().cpu().numpy(), dtype=np.float32)
    if feature_mask is None:
        fmask = np.ones(F, dtype=np.uint8)
    else:
        fmask = np.ascontiguousarray(
            feature_mask.detach().cpu().numpy().astype(bool), dtype=np.uint8)
    use_imat = interaction_matrix is not None
    if use_imat:
        imat = np.ascontiguousarray(
            interaction_matrix.detach().cpu().numpy(), dtype=np.float32)
    else:
        imat = np.zeros((1, 1), dtype=np.float32)

    if sparse_layout is not None:
        indptr = np.ascontiguousarray(sparse_layout[0], dtype=np.int64)
        indices = np.ascontiguousarray(sparse_layout[1], dtype=np.int32)
        data = np.ascontiguousarray(sparse_layout[2], dtype=np.int32)
        default_bin = np.ascontiguousarray(sparse_layout[3], dtype=np.int32)
        use_sparse = 1
    else:
        indptr = np.zeros(1, dtype=np.int64)
        indices = np.zeros(1, dtype=np.int32)
        data = np.zeros(1, dtype=np.int32)
        default_bin = np.zeros(1, dtype=np.int32)
        use_sparse = 0

    # nbins depends only on the (fixed) binner, so cache it on the binner instead
    # of rebuilding the Python generator every round.
    nbins = getattr(binner, "_c_nbins", None)
    if nbins is None or nbins.shape[0] != F:
        nbins = np.fromiter(
            (min(len(e) + 1, MAX_BINS) for e in binner.edges_), dtype=np.int32, count=F)
        binner._c_nbins = nbins

    max_leaves = int(params.max_leaves)
    max_nodes = 2 * max_leaves + 1
    out_feature = np.empty(max_nodes, dtype=np.int64)
    out_thr_bin = np.empty(max_nodes, dtype=np.int64)
    out_left = np.empty(max_nodes, dtype=np.int64)
    out_right = np.empty(max_nodes, dtype=np.int64)
    out_value = np.empty(max_nodes, dtype=np.float32)
    out_depth = np.empty(max_nodes, dtype=np.int64)

    n_nodes = lib.cgrow(
        _ptr(bn), _ptr(gn), _ptr(hn), _ptr(fmask), _ptr(imat),
        ctypes.c_float(float(interaction_boost)), ctypes.c_int(1 if use_imat else 0),
        _ptr(indptr), _ptr(indices), _ptr(data), _ptr(default_bin),
        ctypes.c_int(use_sparse), _ptr(nbins),
        ctypes.c_int64(int(n)), ctypes.c_int(int(F)), ctypes.c_int(int(MAX_BINS)),
        ctypes.c_float(float(params.reg_lambda)), ctypes.c_float(float(params.gamma)),
        ctypes.c_float(float(params.min_child_weight)),
        ctypes.c_int(int(params.min_samples_leaf)),
        ctypes.c_float(float(params.learning_rate)),
        ctypes.c_int(max_leaves), ctypes.c_int(int(params.max_depth)),
        ctypes.c_int(int(n_threads)),
        _ptr(out_feature), _ptr(out_thr_bin), _ptr(out_left), _ptr(out_right),
        _ptr(out_value), _ptr(out_depth),
    )

    feat = out_feature[:n_nodes]
    thr_bin = out_thr_bin[:n_nodes]
    left = out_left[:n_nodes]
    right = out_right[:n_nodes]
    value = out_value[:n_nodes]
    depth = out_depth[:n_nodes]

    scales = binner.scales_.clamp_min(1e-12)
    threshold = np.zeros(n_nodes, dtype=np.float32)
    gate = np.ones(n_nodes, dtype=np.float32)
    for nid in range(n_nodes):
        f = int(feat[nid])
        if f != LEAF:
            threshold[nid] = binner.edge_value(f, int(thr_bin[nid]))
            gate[nid] = float(scales[f])

    return Tree(
        feature=torch.from_numpy(np.ascontiguousarray(feat)).to(dev),
        threshold=torch.from_numpy(threshold).to(dev),
        left=torch.from_numpy(np.ascontiguousarray(left)).to(dev),
        right=torch.from_numpy(np.ascontiguousarray(right)).to(dev),
        value=torch.from_numpy(np.ascontiguousarray(value)).to(dev),
        depth=int(depth.max()) + 1,
        gate_scale=torch.from_numpy(gate).to(dev),
    )


if __name__ == "__main__":  # `python -m yabt.grow_c` -> prebuild + diagnostics
    raise SystemExit(0 if prebuild() else 1)
