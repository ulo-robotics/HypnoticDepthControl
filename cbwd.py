"""
The backward sweep as one CUDA kernel (cuda/cbwd.cu), ported from the old meta_opt engine (cengine.py). Same contract as
kstep.xla_backward(kp, q, st, ftape, T) (blocked S4D adjoint, T steps per block). With grads=True the kernel also returns the per-patient gradients
that are sums over steps and the tapes of the large matrices' gradients, and assemble_tapes turns them into every per-patient gradient without
replaying the steps (kstep.assemble replays every step and takes its VJP again).

Layout of the kernel: one thread block = PB patients of one network, LANES threads per patient, the GLU / head matmuls on tensor cores across the
block's patients. One shared library per shape and launch option, built with nvcc on first use and cached by content hash next to kcuda's.
"""
import ctypes, hashlib, math, os, subprocess
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import model as M
import policy as P
import kstep as K
import kcuda as KC

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "cbwd.cu")
SRC_FWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "cfwd.cu")
SRC_ASM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda", "casm.cu")
SMEM_MAX = 101376                     # dynamic shared memory per block (sm_89)
NQ = 48
ASM_PREC = K.HI                       # precision of the assembly GEMMs
_LOADED = {}


def dims(kp):
    H, NS = kp["layers"][0]["Ar"].shape
    return dict(H=H, NS=NS, L=len(kp["layers"]), HD=(kp["hW"].shape[1] if "hW" in kp else 0), PP=kp["lamr"].shape[0])


def smem_bytes(d, pb):
    H, L, HDP, PP = d["H"], d["L"], max(d["HD"], 8), d["PP"]
    lds = max(2 * H, HDP) + 4
    nv = 7 + 5 * L
    return 4 * (3 * L * pb * H + L * pb * 2 + 3 * pb * lds + nv * H + 2 * HDP + 8 * PP + pb * NQ)


def valid(d, B, pb, lanes, T):
    return (lanes in (4, 8, 16, 32) and d["H"] % lanes == 0 and B % pb == 0 and (pb * lanes) % 32 == 0 and pb * lanes <= 1024 and M.N % T == 0
            and smem_bytes(d, pb) <= SMEM_MAX)


def default_lanes(H):
    return 32 if H >= 64 else 16


def nsm(d):
    return 6 * d["H"] + max(d["HD"], 8) + 8 * d["PP"] + 2 + 2 * d["L"] * d["H"]


# ---------------------------------------------------------------- inputs (one network; under vmap every input gets the member axis)
MM_WEIGHTS = ("head_W1", "W1T", "L_Wv", "L_Wg", "L_WvT", "L_WgT")      # the kernels' tensor-core operands: passed as TF32 hi / lo parts


def tf32_rna(x):
    """x rounded to TF32 (10 mantissa bits) as cvt.rna.tf32.f32: to nearest, ties away from zero"""
    u = lax.bitcast_convert_type(x, jnp.uint32)
    return lax.bitcast_convert_type((u + jnp.uint32(0x1000)) & jnp.uint32(0xFFFFE000), jnp.float32)


def tf32_split(w):
    """w ~ hi + lo, both TF32 (the 3xTF32 operand split, done once instead of per step)"""
    hi = tf32_rna(w)
    return hi, tf32_rna(w - hi)


def pack_q(q):
    B = q["mu"].shape[0]
    cols = [q[f"A{i}"] for i in range(16)] + [q[f"B{i}"] for i in range(4)] + [q[k] for k in ("mu", "stim", "stim_t", "a_ou", "gamma", "e0", "emax", "osc")]
    cols = jnp.stack(cols, 1)
    return jnp.concatenate([cols, jnp.zeros((B, 32 - cols.shape[1]), cols.dtype), q["dmask"]], 1)


def inputs(kp, q, st, ftape, T, tapes=True):
    d = dims(kp)
    H, L, dt = d["H"], d["L"], kp["sc"].dtype
    if "mH" in kp:                                                       # masked layout
        mH, nH, ho, gl, dec_lin, head_dec = kp["mH"], kp["sc"][4], kp["sc"][5], kp["sc"][6:6 + L], kp["decl"], kp["dec"]
    elif "hW" in kp:                                                     # plain, with head
        mH, nH, ho, gl, dec_lin, head_dec = jnp.ones(H, dt), jnp.asarray(H, dt), jnp.asarray(1.0, dt), jnp.ones(L, dt), jnp.zeros(H, dt), kp["dec"]
    else:                                                                # plain, linear decoder
        mH, nH, ho, gl, dec_lin, head_dec = jnp.ones(H, dt), jnp.asarray(H, dt), jnp.asarray(0.0, dt), jnp.ones(L, dt), kp["dec"], None
    z = jnp.zeros((), dt)
    ins = dict(sc=jnp.stack([nH, ho, kp["sc"][0], kp["sc"][1], kp["sc"][2], kp["sc"][3], z, z]), gl=gl, mH=mH, enc_b=kp["enc_b"],
               encW=jnp.stack([kp[f"enc{i}"] for i in range(4)]), dec_lin=dec_lin)
    if d["HD"]:
        ins.update(head_W1=kp["hW"], W1T=kp["hW"].T, head_b1=kp["hb"], head_dec=head_dec)
    ins["p5"] = jnp.stack([kp[k] for k in ("lamr", "lami", "Bur", "Bui", "Bwr", "Bwi", "Cor", "Coi")])
    ly = kp["layers"]
    ins["L_vec"] = jnp.stack([jnp.stack([lp[k] for k in ("D", "bv", "bg", "g", "beta")]) for lp in ly])
    ins["L_Wv"] = jnp.stack([lp["Wv"] for lp in ly])
    ins["L_Wg"] = jnp.stack([lp["Wg"] for lp in ly])
    ins["L_WvT"] = jnp.swapaxes(ins["L_Wv"], 1, 2)
    ins["L_WgT"] = jnp.swapaxes(ins["L_Wg"], 1, 2)
    ins["q_c"] = pack_q(q)
    ins["s_in"] = jnp.stack([st[k] for k in ("z0", "tg", "sy", "r", "rho")])
    ins["b_K"] = jnp.stack([K.block_mats(lp, T)["K"].T for lp in ly])
    ins["b_S4"] = jnp.stack([jnp.stack([lp[k].T for k in ("Ar", "Ai", "Br", "Bi", "Cr", "Ci")]) for lp in ly])    # (L, 6, NS, H): channels fastest
    for k in [k for k in MM_WEIGHTS if k in ins]:
        ins[k], ins[k + "_lo"] = tf32_split(jnp.asarray(ins[k], jnp.float32))
    if tapes:                                                            # the forward kernel's native tapes when there (no repacking)
        ins["c_tape"] = ftape["packed"] if "packed" in ftape else KC.pack_tape(ftape["c"])
        ins["t_ys"] = ftape["ys_all"] if "ys_all" in ftape else jnp.stack(ftape["ys"])
    return {k: jnp.asarray(v, jnp.float32) for k, v in ins.items()}


def outputs(d, B, T, grads, lam):
    N, L, H, NS, HD = M.N, d["L"], d["H"], d["NS"], d["HD"]
    o = dict(lam=(N, B, 6 + 2 * d["PP"] + K.DP), e=(L, N, B, H)) if lam else {}
    o.update(t_x=(B, N, 2 * L * H), s_l=(L, 2, B, NS, H), s_F=(L, T, B, H))
    if grads:
        o.update(g_sm=(B, nsm(d)), t_a=(L, B, N, H), t_dzv=(L, B, N, H), t_dzg=(L, B, N, H), t_xe=(B, N, H))
        if HD:
            o["t_dpre"] = (B, N, HD)
    return o


# ---------------------------------------------------------------- build
def _header(names_in, names_out, d, T, pb, lanes, grads, tc, lam=True):
    f = lambda v: f"{float(v)!r}f"
    lines = [f"#define HP {d['H']}", f"#define NSP {d['NS']}", f"#define LP {d['L']}", f"#define HDP {max(d['HD'], 8)}", f"#define HEAD {int(d['HD'] > 0)}",
             f"#define P5P {d['PP']}", f"#define TBLK {T}", f"#define NSTEP {M.N}", f"#define PB {pb}", f"#define DPAD {K.DP}",
             f"#define LANES {lanes}", f"#define TC {int(tc)}", f"#define GRADS {int(grads)}", f"#define LAMOUT {int(lam)}"]
    for n, v in (("DT_", M.DT), ("E0_", M.E0), ("EMAX_", M.EMAX), ("GAMMA_", M.GAMMA), ("C50N_", M.C50N), ("UMAX_", M.UMAX), ("T_END_", M.T_END)):
        lines.append(f"#define {n} {f(v)}")
    lines.append(f"#define NIN {len(names_in)}")
    lines += [f"#define I_{n} {i}" for i, n in enumerate(names_in)]
    lines.append(f"#define NOUT {len(names_out)}")
    lines += [f"#define O_{n} {i}" for i, n in enumerate(names_out)]
    return "\n".join(lines) + "\n"


def _build(header, src_path=SRC, symbol="CbwdHandler", prefix="cbwd"):
    with KC._BUILD_LOCK:
        if (header, prefix) in _LOADED:
            return _LOADED[(header, prefix)]
        src = open(src_path).read()
        arch = KC._arch()
        tag = hashlib.sha1((header + src + arch).encode()).hexdigest()[:16]
        so = os.path.join(KC.CACHE, f"{prefix}_{tag}.so")
        if not os.path.exists(so):
            os.makedirs(KC.CACHE, exist_ok=True)
            hf = os.path.join(KC.CACHE, f"{prefix}_{tag}.h")
            open(hf, "w").write(header)
            cmd = [KC.nvcc(), "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC", f"-arch={arch}", "-Xlinker", "--exclude-libs,ALL",
                   "-Xcompiler", "-static-libstdc++", "-Xcompiler", "-static-libgcc", "-diag-suppress", "177,550",
                   "-I" + jax.ffi.include_dir(), "-include", hf, src_path, "-o", f"{so}.{os.getpid()}.tmp"]
            if os.environ.get("CBWD_VERBOSE"):
                cmd += ["-Xptxas", "-v"]
            if os.environ.get("CBWD_LINEINFO"):
                cmd += ["-lineinfo"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"nvcc failed:\n{res.stderr[-4000:]}")
            if os.environ.get("CBWD_VERBOSE"):
                print("\n".join(l for l in res.stderr.splitlines() if "registers" in l or "spill" in l), flush=True)
            os.replace(f"{so}.{os.getpid()}.tmp", so)
        lib = ctypes.cdll.LoadLibrary(so)
        name = f"pkpd_{prefix}_{tag}"
        jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(lib, symbol)), platform="CUDA")
        _LOADED[(header, prefix)] = name
        return name


# ---------------------------------------------------------------- the sweep
def backward(kp, q, st, ftape, T=20, pb=16, lanes=None, grads=False, tc=True, lam=True):
    """kstep.xla_backward(kp, q, st, ftape, T) on the CUDA kernel; grads=True: plus the per-patient gradient sums and tapes (for assemble_tapes);
    lam=False: without lam / e (assemble_tapes does not need them)"""
    d = dims(kp)
    B = q["mu"].shape[0]
    lanes = lanes or default_lanes(d["H"])
    if not valid(d, B, pb, lanes, T):
        raise ValueError(f"cbwd: pb={pb} lanes={lanes} T={T} not valid for {d} (shared memory {smem_bytes(d, pb)} B)")
    ins = inputs(kp, q, st, ftape, T)
    outs = outputs(d, B, T, grads, lam)
    name = _build(_header(list(ins), list(outs), d, T, pb, lanes, grads, tc, lam))
    res = jax.ffi.ffi_call(name, [jax.ShapeDtypeStruct(s, jnp.float32) for s in outs.values()], vmap_method="broadcast_all")(
        *ins.values(), B=np.int32(B))
    r = dict(zip(outs, res))
    bt = dict(lam=KC.unpack_tape(r["lam"], d["PP"]), e=[r["e"][l] for l in range(d["L"])]) if lam else {}
    if grads:
        bt.update({k: r[k] for k in ("g_sm", "t_x", "t_a", "t_dzv", "t_dzg", "t_xe", "t_dpre") if k in r})
    return bt


# ---------------------------------------------------------------- the forward sweep (cuda/cfwd.cu, blocked S4D)
def smem_fwd(d, pb, T):
    H, L, HDP, PP = d["H"], d["L"], max(d["HD"], 8), d["PP"]
    return 4 * (3 * pb * (max(2 * H, HDP) + 4) + (7 + 6 * L) * H + 2 * HDP + 8 * PP + L * T * H + pb * NQ)


def forward(kp, q, st, tape=True, T=8, pb=16, lanes=None, tc=True):
    """kstep.xla_forward(kp, q, st, tape, T) on the CUDA kernel: (tape dict or None, per-patient sums)"""
    d = dims(kp)
    B = q["mu"].shape[0]
    lanes = lanes or default_lanes(d["H"])
    if not (lanes in (4, 8, 16, 32) and d["H"] % lanes == 0 and B % pb == 0 and pb % 8 == 0 and (pb * lanes) % 32 == 0 and pb * lanes <= 1024
            and M.N % T == 0 and smem_fwd(d, pb, T) <= SMEM_MAX):
        raise ValueError(f"cfwd: pb={pb} lanes={lanes} T={T} not valid for {d}")
    full = inputs(kp, q, st, dict(c=None, ys=None), T, tapes=False)
    ins = {k: full[k] for k in ("sc", "gl", "mH", "enc_b", "encW", "dec_lin", "head_W1", "head_b1", "head_dec", "p5", "L_vec", "L_Wv", "L_Wg",
                                "q_c", "b_K", "b_S4", "L_Wv_lo", "L_Wg_lo", "head_W1_lo") if k in full}
    ins["s_in"] = jnp.stack([st[k] for k in ("z0", "tg", "sy", "r", "rho", "z1")]).astype(jnp.float32)
    N, L, H, NS = M.N, d["L"], d["H"], d["NS"]
    outs = dict(tape=(N, B, 6 + 2 * d["PP"] + K.DP), ys=(L, N, B, H), sums=(3, B), s_h=(L, 2, B, NS, H))
    name = _build(_header(list(ins), list(outs), d, T, pb, lanes, False, tc, True), SRC_FWD, "CfwdHandler", "cfwd")
    res = jax.ffi.ffi_call(name, [jax.ShapeDtypeStruct(s_, jnp.float32) for s_ in outs.values()], vmap_method="broadcast_all")(
        *ins.values(), B=np.int32(B), wtape=np.int32(1 if tape else 0))
    r = dict(zip(outs, res))
    sums = dict(cost=r["sums"][0], se=r["sums"][1], viol=r["sums"][2])
    if not tape:
        return None, sums
    return dict(c=KC.unpack_tape(r["tape"], d["PP"]), ys=[r["ys"][l] for l in range(L)], packed=r["tape"], ys_all=r["ys"]), sums


# ---------------------------------------------------------------- replay-free assembly
CASM_CK = (30, 24, 20, 12, 10, 8)            # steps per chunk, largest first (measured at H32 NS16 L3: 30 best, 21 -> 18 ms from 12 to 30)


def casm(tp, btape, ck=None, nt=256):
    """cuda/casm.cu on backward(..., grads=True)'s tapes: dict(pq (B, L, H, NS, 4) S4D moments, g_w (B, L, 2, H, H) GLU matrices,
    g_b (B, L, 3, H) bv / bg / D, and with a head g_hw (B, H, HD), g_hb (B, HD)). ck: steps per chunk (None: the largest that fits)"""
    ly = tp["extra"]["layers"]
    L, H, NS = ly["log_A_re"].shape
    HD = btape["t_dpre"].shape[-1] if "t_dpre" in btape else 0
    B = btape["g_sm"].shape[0]
    fits = lambda c: M.N % c == 0 and 4 * c * (2 * L * H + 3 * L * H + H + max(HD, 8)) <= SMEM_MAX
    ck = ck or next((c for c in CASM_CK if fits(c)), None)
    if ck is None or not fits(ck):
        raise ValueError(f"casm: ck={ck} does not fit (H{H} L{L} HD{HD})")
    nm = L * H * NS
    rm = min(6, -(-nm // nt))
    ms = -(-nm // (nt * rm))
    A = -jnp.exp(ly["log_A_re"]) + 1j * ly["A_im"]
    dA = jnp.exp(jnp.exp(ly["log_dt"])[..., None] * A)
    ins = dict(dA=jnp.stack([jnp.real(dA), jnp.imag(dA)], -1), t_x=btape["t_x"], t_a=btape["t_a"], t_dzv=btape["t_dzv"], t_dzg=btape["t_dzg"])
    outs = dict(pq=(B, L, H, NS, 4), g_w=(B, L, 2, H, H), g_b=(B, L, 3, H))
    if HD:
        ins.update(t_xe=btape["t_xe"], t_dpre=btape["t_dpre"])
        outs.update(g_hw=(B, H, HD), g_hb=(B, HD))
    lines = [f"#define HP {H}", f"#define NSP {NS}", f"#define LP {L}", f"#define HDP {max(HD, 8)}", f"#define HEAD {int(HD > 0)}",
             f"#define NSTEP {M.N}", f"#define NT {nt}", f"#define CK {ck}", f"#define RM {rm}", f"#define MS {ms}", f"#define NIN {len(ins)}"]
    lines += [f"#define I_{k} {i}" for i, k in enumerate(ins)] + [f"#define NOUT {len(outs)}"] + [f"#define O_{k} {i}" for i, k in enumerate(outs)]
    name = _build("\n".join(lines) + "\n", SRC_ASM, "CasmHandler", "casm")
    res = jax.ffi.ffi_call(name, [jax.ShapeDtypeStruct(s_, jnp.float32) for s_ in outs.values()], vmap_method="broadcast_all")(
        *[jnp.asarray(v, jnp.float32) for v in ins.values()], B=np.int32(B))
    return dict(zip(outs, res))


def assemble_tapes(tp, sz, w_low, rho_u, plain, q, st, ftape, btape, chunk=None, cuda=False, ck=None):
    """per-patient gradients (as kstep.assemble) from backward(..., grads=True): small parameters from the kernel's sums, the large matrices as
    GEMMs over the steps, the S4D kernels from cuda/s4grad.cu, then the VJP of the kernel layout. The whole batch at once: unlike the replay
    assembly nothing here grows with the batch beyond the tapes themselves (chunking only copied them). chunk: unused (the engine's signature).
    cuda=True: the matrices, sums and S4D moments in one pass over the tapes (cuda/casm.cu, ck steps per chunk)"""
    B = q["mu"].shape[0]
    kp, kp_vjp = jax.vjp(lambda t: K.to_kp(t, sz, w_low, rho_u, plain), tp)
    mS = P.masks(sz, P.shape_of(tp), tp["phys"]["u0"].dtype)["mS"]
    d = dims(kp)
    H, L, HD, PP = d["H"], d["L"], d["HD"], d["PP"]
    HDP = max(HD, 8)
    ein = lambda a, b: jnp.einsum("bkh,bko->bho", a, b, precision=ASM_PREC)
    gsm, X = btape["g_sm"], btape["t_x"]
    g = jax.tree_util.tree_map(lambda a: jnp.zeros((B,) + a.shape, a.dtype), kp)
    o = 0

    def take(n):
        nonlocal o
        v = gsm[:, o:o + n]
        o += n
        return v
    for i in range(4):
        g[f"enc{i}"] = take(H)
    g["enc_b"] = take(H)
    decl = take(H)
    hdec = take(HDP)[:, :HD]
    if "mH" in kp:
        g["decl"], g["dec"] = decl, hdec
    elif "hW" in kp:
        g["dec"] = hdec
    else:
        g["dec"] = decl
    for k in ("lamr", "lami", "Bur", "Bui", "Bwr", "Bwi", "Cor", "Coi"):
        g[k] = take(PP)
    decb, u0 = take(1)[:, 0], take(1)[:, 0]
    g["sc"] = g["sc"].at[:, 0].set(u0).at[:, 1].set(decb)
    lng, lnb = take(L * H).reshape(B, L, H), take(L * H).reshape(B, L, H)
    r = casm(tp, btape, ck) if cuda else None
    for l in range(L):
        gl = g["layers"][l]
        gl["g"], gl["beta"] = lng[:, l], lnb[:, l]
        if cuda:
            gl["Wv"], gl["Wg"] = r["g_w"][:, l, 0], r["g_w"][:, l, 1]
            gl["bv"], gl["bg"], gl["D"] = r["g_b"][:, l, 0], r["g_b"][:, l, 1], r["g_b"][:, l, 2]
            continue
        aa, dzv, dzg = btape["t_a"][l], btape["t_dzv"][l], btape["t_dzg"][l]
        gl["Wv"], gl["Wg"] = ein(aa, dzv), ein(aa, dzg)
        gl["bv"], gl["bg"] = dzv.sum(1), dzg.sum(1)
        gl["D"] = jnp.einsum("bkh,bkh->bh", X[..., (2 * l + 1) * H:(2 * l + 2) * H], X[..., 2 * l * H:(2 * l + 1) * H], precision=K.HI)
    if HD:
        if cuda:
            g["hW"], g["hb"] = r["g_hw"], r["g_hb"]
        else:
            dpre = btape["t_dpre"]
            g["hW"], g["hb"] = ein(btape["t_xe"], dpre), dpre.sum(1)
    gt = jax.vmap(lambda gk: kp_vjp(gk)[0])(g)
    glt = gt["extra"]["layers"]
    s4 = KC.s4d_from_moments(tp, mS, r["pq"]) if cuda else KC.s4d_grads(tp, mS, X, 0, H, 2 * H)
    for k, v in s4.items():
        glt[k] = glt[k] + v
    return gt


def grads(tp, sz, w_low, rho_u, plain, q, st, ftape, T=10, pb=16, lanes=None, chunk=None, casm=False):
    """backward with tapes + replay-free assembly in one: the engine's joint candidate (the assembly signature with the kernel's own backward)"""
    kp = K.to_kp(tp, sz, w_low, rho_u, plain)
    bt = backward(kp, q, st, ftape, T, pb=pb, lanes=lanes, grads=True, lam=False)
    return assemble_tapes(tp, sz, w_low, rho_u, plain, q, st, ftape, bt, chunk, cuda=casm)
