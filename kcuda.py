"""
CUDA implementation of the two sweeps of kstep.py (cuda/sweep.cu), called as XLA FFI custom calls. One shared library per shape
(H, NS, L, head, padded physics modes) and launch option (warps per block, weight staging in shared memory), built with nvcc on first use and cached by content hash in $PKPD_CACHE/cuda
(default ~/.cache/pkpd_v1/cuda, a few seconds per build). available() says whether nvcc, the XLA FFI headers and a CUDA device are there.
"""
import ctypes, hashlib, os, shutil, subprocess, threading
import numpy as np
import jax
import jax.numpy as jnp

import model as M
import kstep as K

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "sweep.cu")
CACHE = os.path.join(os.environ.get("PKPD_CACHE") or os.path.expanduser("~/.cache/pkpd_v1"), "cuda")
_LOADED = {}
_BUILD_LOCK = threading.Lock()                # engine.py compiles candidates in parallel threads


def nvcc():
    p = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    return p if os.path.exists(p) else None


def available():
    try:
        return nvcc() is not None and jax.default_backend() == "gpu" and hasattr(jax, "ffi") and os.path.exists(SRC)
    except Exception:
        return False


def _arch():
    cc = getattr(jax.devices()[0], "compute_capability", None)
    if cc is None:
        cc = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()[0]
    return "sm_" + str(cc).replace(".", "")


def dims(kp):
    H, NS = kp["layers"][0]["Ar"].shape
    return dict(H=H, NS=NS, NL=len(kp["layers"]), HD=(kp["hW"].shape[1] if "hW" in kp else 0), PP=kp["lamr"].shape[0])


def build(d, wpb, stage=0):
    """compile (or load from the cache) the library for these sizes and launch options; returns the FFI target names"""
    key = tuple(sorted(d.items())) + (("WPB", wpb), ("STAGE", int(stage)))
    with _BUILD_LOCK:
        if key not in _LOADED:
            _LOADED[key] = _build(key)
        return _LOADED[key]


def _build(key):
    src = open(SRC).read()
    arch = _arch()
    defs = [f"-D{k}={v}" for k, v in key]
    tag = hashlib.sha1((src + " ".join(defs) + arch).encode()).hexdigest()[:16]
    so = os.path.join(CACHE, f"sweep_{tag}.so")
    if not os.path.exists(so):
        os.makedirs(CACHE, exist_ok=True)
        cmd = [nvcc(), "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", f"-arch={arch}",
               "-Xlinker", "--exclude-libs,ALL", "-Xcompiler", "-static-libstdc++", "-Xcompiler", "-static-libgcc",   # no libstdc++ version clash with the host
               "-I" + jax.ffi.include_dir(), *defs, SRC, "-o", f"{so}.{os.getpid()}.tmp"]      # per-process temporary file:
        res = subprocess.run(cmd, capture_output=True, text=True)                             # several processes may build at once
        if res.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{res.stderr[-3000:]}")
        os.replace(f"{so}.{os.getpid()}.tmp", so)
    lib = ctypes.cdll.LoadLibrary(so)
    names = (f"pkpd_fwd_{tag}", f"pkpd_bwd_{tag}")
    jax.ffi.register_ffi_target(names[0], jax.ffi.pycapsule(lib.PkpdFwd), platform="CUDA")
    jax.ffi.register_ffi_target(names[1], jax.ffi.pycapsule(lib.PkpdBwd), platform="CUDA")
    return names


# ---------------------------------------------------------------- layouts (must match cuda/sweep.cu)
def pack_kp(kp):
    parts = [kp["sc"]] + [kp[k] for k in ("lamr", "lami", "Bur", "Bui", "Bwr", "Bwi", "Cor", "Coi", "enc0", "enc1", "enc2", "enc3", "enc_b", "dec")]
    if "hW" in kp:
        parts += [kp["hW"].ravel(), kp["hW"].T.ravel(), kp["hb"]]
    for lp in kp["layers"]:
        parts += [lp[k].ravel() for k in ("Ar", "Ai", "Br", "Bi", "Cr", "Ci")] + [lp["K0"], lp["D"]]
        parts += [lp["Wv"].ravel(), lp["Wv"].T.ravel(), lp["Wg"].ravel(), lp["Wg"].T.ravel(), lp["bv"], lp["bg"], lp["g"], lp["beta"]]
    return jnp.concatenate([jnp.ravel(p) for p in parts]).astype(jnp.float32)


def pack_q(q):
    cols = [q[k][:, None] for k in ("mu", "a_ou", "osc", "stim", "gamma", "e0", "emax")] + [q["dmask"]]
    cols += [q[f"A{i}"][:, None] for i in range(16)] + [q[f"B{i}"][:, None] for i in range(4)]
    return jnp.concatenate(cols, 1).astype(jnp.float32)


def pack_st(st):
    return jnp.stack([st[k] for k in K.STEP_IN]).astype(jnp.float32)


def pack_tape(c):
    return jnp.concatenate([c[k][..., None] for k in ("x0", "x1", "x2", "x3", "ou", "up")] + [c["hpr"], c["hpi"], c["buf"]], -1)


def unpack_tape(t, PP):
    out = {k: t[..., i] for i, k in enumerate(("x0", "x1", "x2", "x3", "ou", "up"))}
    out.update(hpr=t[..., 6:6 + PP], hpi=t[..., 6 + PP:6 + 2 * PP], buf=t[..., 6 + 2 * PP:])
    return out


# ---------------------------------------------------------------- sweeps (same contract as kstep.xla_forward / xla_backward)
def forward(kp, q, st, tape=True, wpb=4, stage=0):
    d = dims(kp)
    B, S = q["mu"].shape[0], 6 + 2 * d["PP"] + K.DP
    assert B % wpb == 0, (B, wpb)
    fwd, _ = build(d, wpb, stage)
    shapes = [jax.ShapeDtypeStruct((M.N, B, S) if tape else (1,), jnp.float32),
              jax.ShapeDtypeStruct((d["NL"], M.N, B, d["H"]) if tape else (1,), jnp.float32), jax.ShapeDtypeStruct((3, B), jnp.float32)]
    t, ys, sums = jax.ffi.ffi_call(fwd, shapes)(pack_kp(kp), pack_q(q), pack_st(st), tape=np.int32(1 if tape else 0))
    s = dict(cost=sums[0], se=sums[1], viol=sums[2])
    return (dict(c=unpack_tape(t, d["PP"]), ys=[ys[l] for l in range(d["NL"])]) if tape else None), s


def backward(kp, q, st, ftape, wpb=4, stage=0):
    d = dims(kp)
    B, S = q["mu"].shape[0], 6 + 2 * d["PP"] + K.DP
    assert B % wpb == 0, (B, wpb)
    _, bwd = build(d, wpb, stage)
    shapes = [jax.ShapeDtypeStruct((M.N, B, S), jnp.float32), jax.ShapeDtypeStruct((d["NL"], M.N, B, d["H"]), jnp.float32)]
    lam, e = jax.ffi.ffi_call(bwd, shapes)(pack_kp(kp), pack_q(q), pack_st(st), pack_tape(ftape["c"]), jnp.stack(ftape["ys"]))
    return dict(lam=unpack_tape(lam, d["PP"]), e=[e[l] for l in range(d["NL"])])
