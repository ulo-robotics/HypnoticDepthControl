"""
CUDA support shared by the blocked engine (cbwd.py): the build registry (one nvcc build per library and process, in parallel across
libraries; the .so files cached by content hash in $PKPD_CACHE/cuda, default ~/.cache/pkpd_v1/cuda), the packing of the small-carry tape,
and the S4D-parameter gradients from the moments that cuda/casm.cu assembles. available() says whether nvcc, the XLA FFI headers and a CUDA
device are there.
"""
import os, shutil, subprocess, threading
import jax
import jax.numpy as jnp
from jax import lax

import model as M

CACHE = os.path.join(os.environ.get("PKPD_CACHE") or os.path.expanduser("~/.cache/pkpd_v1"), "cuda")
_LOADED = {}
_BUILD_LOCK = threading.Lock()                # the registry (_LOADED, _INFLIGHT, FFI registration): engine.py builds candidates in parallel threads
_INFLIGHT = {}


def once(key, build):
    """build() for key once per process: concurrent callers of the same key wait for the first one, different keys build in parallel (nvcc is
    a process of its own; only the registry is locked)"""
    with _BUILD_LOCK:
        if key in _LOADED:
            return _LOADED[key]
        ev = _INFLIGHT.get(key)
        first = ev is None
        if first:
            ev = _INFLIGHT[key] = threading.Event()
    if not first:
        ev.wait()
        with _BUILD_LOCK:
            if key in _LOADED:
                return _LOADED[key]
        raise RuntimeError(f"build of {key[:2]} failed in another thread")
    try:
        val = build()
        with _BUILD_LOCK:
            _LOADED[key] = val
        return val
    finally:
        with _BUILD_LOCK:
            _INFLIGHT.pop(key, None)
        ev.set()


def register(name, capsule):
    with _BUILD_LOCK:
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA")


def nvcc():
    p = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    return p if os.path.exists(p) else None


def available():
    try:
        return nvcc() is not None and jax.default_backend() == "gpu" and hasattr(jax, "ffi")
    except Exception:
        return False


def _arch():
    cc = getattr(jax.devices()[0], "compute_capability", None)
    if cc is None:
        cc = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()[0]
    return "sm_" + str(cc).replace(".", "")


def pack_tape(c):
    return jnp.concatenate([c[k][..., None] for k in ("x0", "x1", "x2", "x3", "ou", "up")] + [c["hpr"], c["hpi"], c["buf"]], -1)


def unpack_tape(t, PP):
    out = {k: t[..., i] for i, k in enumerate(("x0", "x1", "x2", "x3", "ou", "up"))}
    out.update(hpr=t[..., 6:6 + PP], hpi=t[..., 6 + PP:6 + 2 * PP], buf=t[..., 6 + 2 * PP:])
    return out


def s4d_from_moments(tp, mS, pq):
    """per-patient gradients of log_dt, log_A_re, A_im, C from the moments (B, L, H, NS, 4): Re P, Im P, Re P', Im P'.
    The S4D kernel's contribution is 2 Re sum_n C dB P(dA) with P(dA) known to first order from the moments (P, P'): its gradient is the
    gradient of the surrogate 2 Re sum_n C dB (P + P' (dA - dA_0)) at the current parameters, taken by JAX through the same discretization as
    the reference (model.s4d_A: time constants bounded), so the chain rule through the bound cannot drift from it"""
    ly = tp["extra"]["layers"]
    p0 = {k: ly[k] for k in ("log_dt", "log_A_re", "A_im", "C")}

    def surrogate(p, Pz, Pd):
        dt, A = M.s4d_A(p["log_dt"], p["log_A_re"], p["A_im"])
        dA = jnp.exp(dt * A)
        dB = (dA - 1) / A * mS
        C = p["C"][..., 0] + 1j * p["C"][..., 1]
        return jnp.sum(2 * jnp.real(C * dB * (Pz + Pd * (dA - lax.stop_gradient(dA)))))

    Pz, Pd = pq[..., 0] + 1j * pq[..., 1], pq[..., 2] + 1j * pq[..., 3]
    return jax.vmap(jax.grad(surrogate), in_axes=(None, 0, 0))(p0, Pz, Pd)
