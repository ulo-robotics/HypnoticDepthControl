"""
CUDA implementation of the two sweeps of kstep.py (cuda/sweep.cu), called as XLA FFI custom calls. One shared library per shape
(H, NS, L, head, padded physics modes) and launch option (warps per block, weight staging in shared memory), built with nvcc on first use and cached by content hash in $PKPD_CACHE/cuda
(default ~/.cache/pkpd_v1/cuda, a few seconds per build). available() says whether nvcc, the XLA FFI headers and a CUDA device are there.
"""
import ctypes, hashlib, math, os, shutil, subprocess, threading
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import model as M
import policy as P
import kstep as K

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "sweep.cu")
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
    return once(key, lambda: _build(key))


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
    register(names[0], jax.ffi.pycapsule(lib.PkpdFwd))
    register(names[1], jax.ffi.pycapsule(lib.PkpdBwd))
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
    cols += [q[f"A{i}"][:, None] for i in range(16)] + [q[f"B{i}"][:, None] for i in range(4)] + [q["stim_t"][:, None]]
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


# ---------------------------------------------------------------- gradient assembly (cuda/asm.cu): same contract as kstep.assemble
ASM_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "asm.cu")
NSC = 16


def build_asm(d, wpb, nseg, stage=0):
    key = tuple(sorted(d.items())) + (("WPB", wpb), ("NSEG", nseg), ("STAGE", int(stage)))
    return once(("asm",) + key, lambda: _build_src(ASM_SRC, key, ("PkpdAsm",), "asm")[0])


def _build_src(src_path, key, symbols, prefix):
    src = open(src_path).read()
    arch = _arch()
    defs = [f"-D{k}={v}" for k, v in key]
    tag = hashlib.sha1((src + " ".join(defs) + arch).encode()).hexdigest()[:16]
    so = os.path.join(CACHE, f"{prefix}_{tag}.so")
    if not os.path.exists(so):
        os.makedirs(CACHE, exist_ok=True)
        cmd = [nvcc(), "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", f"-arch={arch}",
               "-Xlinker", "--exclude-libs,ALL", "-Xcompiler", "-static-libstdc++", "-Xcompiler", "-static-libgcc",
               "-I" + jax.ffi.include_dir(), *defs, src_path, "-o", f"{so}.{os.getpid()}.tmp"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{res.stderr[-3000:]}")
        os.replace(f"{so}.{os.getpid()}.tmp", so)
    lib = ctypes.cdll.LoadLibrary(so)
    names = []
    for sym in symbols:
        name = f"pkpd_{sym.lower()}_{tag}"
        register(name, jax.ffi.pycapsule(getattr(lib, sym)))
        names.append(name)
    return names


def asm_dims(kp):
    H, NS = kp["layers"][0]["Ar"].shape
    return dict(H=H, NS=NS, NL=len(kp["layers"]), HD=(kp["hW"].shape[1] if "hW" in kp else 0), PP=kp["lamr"].shape[0])


def pack_kp_asm(kp):
    """kernel parameters in the layout of cuda/asm.cu (masked or plain layout)"""
    d = asm_dims(kp)
    H, L, dt = d["H"], d["NL"], kp["sc"].dtype
    if "mH" in kp:                                       # masked: sc = u0 dec_b w_low rho_u nH head gates
        sc = kp["sc"]
        mH, decl = kp["mH"], kp["decl"]
    else:
        sc = jnp.concatenate([kp["sc"][:4], jnp.asarray([H, 1.0 if "hW" in kp else 0.0], dt), jnp.ones((L,), dt)])
        mH, decl = jnp.ones((H,), dt), jnp.zeros((H,), dt)
    sc = jnp.pad(sc, (0, NSC - sc.shape[0]))
    parts = [sc] + [kp[k] for k in ("lamr", "lami", "Bur", "Bui", "Bwr", "Bwi", "Cor", "Coi", "enc0", "enc1", "enc2", "enc3", "enc_b")]
    parts += [mH, decl, kp["dec"]]
    if "hW" in kp:
        parts += [kp["hW"].ravel(), kp["hW"].T.ravel(), kp["hb"]]
    for lp in kp["layers"]:
        parts += [lp["K0"], lp["D"], lp["Wv"].ravel(), lp["Wv"].T.ravel(), lp["Wg"].ravel(), lp["Wg"].T.ravel(), lp["bv"], lp["bg"], lp["g"], lp["beta"]]
    return jnp.concatenate([jnp.ravel(p) for p in parts]).astype(jnp.float32)


def asm_sizes(d):
    H, HD, PP, L = d["H"], d["HD"], d["PP"], d["NL"]
    HDV = HD if HD > 0 else H
    nsm = 2 + 8 * PP + 4 * H + H + H + HDV + HD + L * 5 * H
    nop = L * 5 * H + H + HD
    return nsm, nop


def asm_call(kp, q, st, c, ys, lam, e, wpb=4, nseg=12, stage=0):
    """one network, one chunk of patients: (small-parameter cotangents (B, NSM), operands (B, N, NOP))"""
    d = asm_dims(kp)
    B = q["mu"].shape[0]
    nsm, nop = asm_sizes(d)
    name = build_asm(d, wpb, nseg, stage)
    shapes = [jax.ShapeDtypeStruct((nseg, B, nsm), jnp.float32), jax.ShapeDtypeStruct((B, M.N, nop), jnp.float32)]
    gs, ops = jax.ffi.ffi_call(name, shapes, vmap_method="broadcast_all")(
        pack_kp_asm(kp), pack_q(q), pack_st(st), pack_tape(c), jnp.stack(ys), pack_tape(lam), jnp.stack(e))
    return gs.sum(0), ops


def g_kp_from(kp, gs, ops):
    """per-patient cotangent of every kernel-layout parameter (leading batch axis) from the kernel's outputs; S4D matrices: zero (their gradient
    comes from the FFT cross-correlation, as in kstep.assemble)"""
    d = asm_dims(kp)
    H, HD, PP, L = d["H"], d["HD"], d["PP"], d["NL"]
    HDV = HD if HD > 0 else H
    B = gs.shape[0]
    g = jax.tree_util.tree_map(lambda a: jnp.zeros((B,) + a.shape, a.dtype), kp)
    o = 0

    def take(n):
        nonlocal o
        v = gs[:, o:o + n]
        o += n
        return v
    sc = jnp.zeros((B, kp["sc"].shape[0]), gs.dtype)
    g["sc"] = sc.at[:, :2].set(take(2))
    for k in ("lamr", "lami", "Bur", "Bui", "Bwr", "Bwi", "Cor", "Coi"):
        g[k] = take(PP)
    for k in ("enc0", "enc1", "enc2", "enc3", "enc_b"):
        g[k] = take(H)
    decl = take(H)
    if "decl" in kp:
        g["decl"] = decl
    g["dec"] = take(HDV)
    hb = take(HD)
    if "hW" in kp:
        g["hb"] = hb
        g["hW"] = jnp.einsum("bkh,bko->bho", ops[..., L * 5 * H:L * 5 * H + H], ops[..., L * 5 * H + H:], precision=K.HI)
    for l in range(L):
        gl = g["layers"][l]
        for k in ("D", "bv", "bg", "g", "beta"):
            gl[k] = take(H)
        a, zvb, zgb = (ops[..., (l * 5 + i) * H:(l * 5 + i + 1) * H] for i in range(3))
        gl["Wv"] = jnp.einsum("bkh,bko->bho", a, zvb, precision=K.HI)
        gl["Wg"] = jnp.einsum("bkh,bko->bho", a, zgb, precision=K.HI)
    return g


def assemble(tp, sz, w_low, rho_u, plain, q, st, ftape, btape, chunk=None, wpb=4, nseg=12, stage=0):
    """per-patient gradients from the two tapes: the CUDA assembly kernel, then batched GEMMs for the large matrices, the FFT cross-correlation
    for the S4D kernels and the VJP of the kernel layout (as kstep.assemble)"""
    B = q["mu"].shape[0]
    kp, kp_vjp = jax.vjp(lambda t: K.to_kp(t, sz, w_low, rho_u, plain), tp)
    mS = P.masks(sz, P.shape_of(tp), tp["phys"]["u0"].dtype)["mS"]
    d = asm_dims(kp)
    L, H = d["NL"], d["H"]
    _, nop = asm_sizes(d)
    if chunk is None:                                    # operands of a chunk: chunk x N x NOP floats per member (~ 128 MB)
        chunk = max(wpb, 1 << int(math.log2(max(1, (32 << 20) // (M.N * nop)))))
    chunk = next(c for c in (chunk, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1) if c <= B and B % c == 0 and (c * nseg) % wpb == 0)

    def chunk_fn(i):
        sl = lambda a, ax: lax.dynamic_slice_in_dim(a, i * chunk, chunk, ax)
        gs, ops = asm_call(kp, {n: sl(v, 0) for n, v in q.items()}, {n: sl(v, 1) for n, v in st.items()},
                           {n: sl(v, 1) for n, v in ftape["c"].items()}, [sl(y, 1) for y in ftape["ys"]],
                           {n: sl(v, 1) for n, v in btape["lam"].items()}, [sl(x, 1) for x in btape["e"]], wpb, nseg, stage)
        g = jax.vmap(lambda gk: kp_vjp(gk)[0])(g_kp_from(kp, gs, ops))
        gl = g["extra"]["layers"]
        for k, v in s4d_grads(tp, mS, ops, 3 * H, 4 * H, 5 * H).items():          # S4D kernels
            gl[k] = gl[k] + v
        return g

    out = lax.map(chunk_fn, jnp.arange(B // chunk))
    return jax.tree_util.tree_map(lambda a: a.reshape((B,) + a.shape[2:]), out)


# ---------------------------------------------------------------- S4D-kernel gradients (cuda/s4grad.cu), for both assemblies
S4_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "s4grad.cu")


def build_s4grad(H, NS, NL):
    cg = min(H, max(1, 256 // NS))
    key = (("CG", cg), ("H", H), ("NL", NL), ("NS", NS))
    return once(("s4",) + key, lambda: _build_src(S4_SRC, key, ("PkpdS4grad",), "s4grad")[0])


def s4d_grads(tp, mS, X, ou, oc, lstride):
    """per-patient gradients of every layer's log_dt, log_A_re, A_im, C (leading batch axis, layer axis second) from the S4D inputs u and output
    cotangents cy of all steps: X (B, N, W), layer l's u at column ou + l * lstride, its cy at oc + l * lstride"""
    ly = tp["extra"]["layers"]
    L, H, NS = ly["log_A_re"].shape
    A = -jnp.exp(ly["log_A_re"]) + 1j * ly["A_im"]
    dlt = jnp.exp(ly["log_dt"])[..., None]
    dA = jnp.exp(dlt * A)
    name = build_s4grad(H, NS, L)
    pq = jax.ffi.ffi_call(name, jax.ShapeDtypeStruct((X.shape[0], L, H, NS, 4), jnp.float32), vmap_method="broadcast_all")(
        jnp.stack([jnp.real(dA), jnp.imag(dA)], -1).astype(jnp.float32), X.astype(jnp.float32),
        ou=np.int32(ou), oc=np.int32(oc), lstride=np.int32(lstride))
    return s4d_from_moments(tp, mS, pq)


def s4d_from_moments(tp, mS, pq):
    """per-patient gradients of log_dt, log_A_re, A_im, C from the moments (B, L, H, NS, 4): Re P, Im P, Re P', Im P'"""
    ly = tp["extra"]["layers"]
    A = -jnp.exp(ly["log_A_re"]) + 1j * ly["A_im"]
    dlt = jnp.exp(ly["log_dt"])[..., None]
    dA = jnp.exp(dlt * A)
    Pz, Pd = pq[..., 0] + 1j * pq[..., 1], pq[..., 2] + 1j * pq[..., 3]
    # k[m] = 2 Re sum_n C dB dA^m  ->  loss = 2 Re F,  F = sum_n C dB P(dA)
    dB = (dA - 1) / A * mS
    C = ly["C"][..., 0] + 1j * ly["C"][..., 1]
    gCc = dB * Pz
    dBdA = (dlt * dA * A - (dA - 1)) / (A * A) * mS
    dFdA = C * (dBdA * Pz + dB * Pd * dlt * dA)
    dFdD = C * (dA * mS * Pz + dB * Pd * A * dA)
    return dict(log_dt=jnp.sum(2 * jnp.real(dFdD), -1) * dlt[..., 0], log_A_re=2 * jnp.real(dFdA * (-jnp.exp(ly["log_A_re"]))),
                A_im=-2 * jnp.imag(dFdA), C=jnp.stack([2 * jnp.real(gCc), -2 * jnp.imag(gCc)], -1))
