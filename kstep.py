"""
The closed-loop step in "kernel form" and the adjoint gradient that the fast training engines share.

Kernel form = the same math as policy.act + the plant of policy.rollout, written so it also traces inside a Pallas (Triton) kernel: real / imaginary
planes instead of complex numbers, no slicing / concatenation / gathers (the GLU halves are separate matrices, the 4x4 plant is 16 scalars, the delay
line shift is a 0/1 matrix), power-of-two shapes (the physics modes are zero-padded), scalars in one vector `sc`.

Adjoint gradient (per patient, exact):
  forward sweep    run the episode, write a TAPE per step: the small carry before the step (plant, drift, physics modes, delay line, last infusion)
                   and the S4D outputs ys of every layer. The big S4D states never go to memory.
  backward sweep   k = N-1 .. 0: replay step k from the tape with the S4D outputs taken from the tape, take its VJP with respect to the small carry
                   and the S4D outputs, and carry two adjoints: of the small carry, and of every S4D state (lam_h, propagated by the transposed S4D
                   recurrence). Writes per step the incoming small-carry adjoint and e = the S4D-recurrence part of the cotangent of each S4D input.
                   No parameter gradient is accumulated here.
  assembly (XLA)   with both tapes, every step is independent: one batched VJP of the replayed step over all (step, patient) pairs gives every
                   per-patient parameter gradient except the S4D kernels (dt, A, C), which follow from an FFT cross-correlation of the S4D input
                   with the S4D output cotangent (the S4D layer is a convolution with K_m = 2 Re sum C dA^m dB).
The sweeps are implemented twice (XLA scans here = the reference, and the blocked CUDA kernels of cbwd.py); `assemble` is the reference assembly.

Parameters come in the training layout of policy.py with the network's sizes sz. A padded network (sizes smaller than the arrays) gets a "masked"
kernel layout: the S4D input mask is folded into dB, and the layer-norm mask (kp["mH"], count sc[4]), the head switch (sc[5]) and the layer gates
(sc[6 + l]) are applied in the step. An unpadded network can use the "plain" layout (no masks).
"""
import math
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import model as M
import policy as P

DP = M.DMAX_ALL + 1              # delay line length (16)
f32 = jnp.float32
HI = lax.Precision.HIGHEST
SMALL = ("x0", "x1", "x2", "x3", "ou", "up", "hpr", "hpi", "buf")     # the small carry (tape) leaves
STEP_IN = ("z0", "z1", "tg", "sy", "r", "rho", "w")                     # per-step inputs, (N, B) each (w: the cost weight, N [k < len] / len)


def p2(n):
    return 1 << max(0, int(n) - 1).bit_length()


def mm(a, W):
    """(.., K) x (K, N): tl.dot / GEMM when the shapes allow it (Triton needs >= 16 everywhere), else a broadcast-reduce"""
    if a.ndim == 2 and min(a.shape[0], a.shape[1], W.shape[1]) >= 16:
        return jnp.dot(a, W, precision=HI, preferred_element_type=a.dtype)
    return jnp.sum(a[..., :, None] * W, -2)


def gelu(x):
    return jax.nn.gelu(x)          # tanh approximation, as policy.act


# ---------------------------------------------------------------- layouts
def kp_dims(tp):
    s = P.shape_of(tp)
    return dict(H=s["H"], NS=s["NS"], L=s["L"], HD=s["HD"], P=s["P"], Pp=max(8, p2(s["P"])))


def s4d_mats(lp, mS=None):
    """discretized S4D (dA, dB, C) as planes, and the zero-lag tap K0 = 2 Re sum C dB; mS masks the inputs of padded states"""
    dt, A = M.s4d_A(lp["log_dt"], lp["log_A_re"], lp["A_im"])
    dA = jnp.exp(dt * A)
    dB = (dA - 1) / A if mS is None else (dA - 1) / A * mS
    C = lp["C"][..., 0] + 1j * lp["C"][..., 1]
    return dict(Ar=jnp.real(dA), Ai=jnp.imag(dA), Br=jnp.real(dB), Bi=jnp.imag(dB), Cr=jnp.real(C), Ci=jnp.imag(C),
                K0=2 * jnp.sum(jnp.real(C * dB), -1))


def to_kp(tp, sz, w_low, rho_u, plain=None):
    """training-layout params of a network of sizes sz -> kernel layout (differentiable).
    plain = None: masked layout; True / False: plain layout with / without the head (only for an unpadded network)"""
    d = kp_dims(tp)
    ph, ex, ly = tp["phys"], tp["extra"], tp["extra"]["layers"]
    dt = ph["u0"].dtype
    mk = P.masks(sz, P.shape_of(tp), dt)
    pad = lambda v: jnp.pad(v, (0, d["Pp"] - d["P"]))
    mag = jnp.exp(-jnp.exp(ph["a"]))
    sc = [ph["u0"][0], ex["dec_b"][0], jnp.asarray(w_low, dt), jnp.asarray(rho_u, dt)]
    kp = dict(lamr=pad(mag * jnp.cos(ph["b"])), lami=pad(mag * jnp.sin(ph["b"])),
              Bur=pad(ph["Bu"][:, 0]), Bui=pad(ph["Bu"][:, 1]), Bwr=pad(ph["Bw"][:, 0]), Bwi=pad(ph["Bw"][:, 1]),
              Cor=pad(ph["Cout"][:, 0]), Coi=pad(ph["Cout"][:, 1]),
              enc0=ex["enc_W"][0], enc1=ex["enc_W"][1], enc2=ex["enc_W"][2], enc3=ex["enc_W"][3], enc_b=ex["enc_b"], layers=[])
    if plain is None:
        kp.update(hW=ex["hW"], hb=ex["hb"], dec=ex["hdec"][:, 0], decl=ex["dec_lin"][:, 0], mH=mk["mH"])
        sc += [mk["nH"], mk["head"].astype(dt)] + [mk["gate"][l].astype(dt) for l in range(d["L"])]
    elif plain:
        kp.update(hW=ex["hW"], hb=ex["hb"], dec=ex["hdec"][:, 0])
    else:
        kp["dec"] = ex["dec_lin"][:, 0]
    kp["sc"] = jnp.stack(sc)
    for l in range(d["L"]):
        lp = {k: v[l] for k, v in ly.items()}
        kp["layers"].append(dict(**s4d_mats(lp, mk["mS"] if plain is None else None), D=lp["D"], Wv=lp["Wv"], Wg=lp["Wg"], bv=lp["bv"],
                                 bg=lp["bg"], g=lp["ln_g"], beta=lp["ln_b"]))
    return kp


def to_kq(pts):
    """patient batch (model.sample_patients) -> per-patient constants (B, ...) and per-step inputs (N, B)"""
    B = pts["mu"].shape[0]
    a = jnp.exp(-M.DT / pts["tau"])
    q = dict(mu=pts["mu"], a_ou=a, osc=pts["sd"] * jnp.sqrt(1 - a * a), stim=pts["stim"], stim_t=pts["stim_t"], stim2=pts["stim2"], stim_t2=pts["stim_t2"], stim3=pts["stim3"], stim_t3=pts["stim_t3"], gamma=pts["gamma"], e0=pts["e0"], emax=pts["emax"],
             dmask=jax.nn.one_hot(pts["delay"], DP, dtype=pts["mu"].dtype))
    Ad = pts["Ad"].reshape(B, 16)
    for i in range(16):
        q[f"A{i}"] = Ad[:, i]
    for i in range(4):
        q[f"B{i}"] = pts["Bd"][:, i]
    wr = lambda t: M.cref(M.C50N, M.GAMMA, M.E0, M.EMAX, t)
    tg = pts["tgt"].T
    tgp = jnp.concatenate([tg[:1], tg[:-1]], 0)
    if "len" in pts:                                                    # variable-length episodes: each patient's own steps, its mean cost
        on = jnp.arange(M.N)[:, None] < pts["len"][None, :]
        w = jnp.where(on, M.N / pts["len"][None, :].astype(tg.dtype), 0.0).astype(tg.dtype)
    else:
        w = jnp.ones_like(tg)
    st = dict(z0=pts["z"][:, 0], z1=pts["z"][:, 1], tg=tg, sy=pts["sy"].T, r=wr(tg) / M.C50N, rho=wr(tgp) / wr(tg), w=w)
    return q, st


def carry0(B, Pp, dtype=f32):
    z = jnp.zeros((B,), dtype)
    return dict(x0=z, x1=z, x2=z, x3=z, ou=z, up=z, hpr=jnp.zeros((B, Pp), dtype), hpi=jnp.zeros((B, Pp), dtype),
                buf=jnp.full((B, DP), M.E0, dtype))


# ---------------------------------------------------------------- the step
@jax.custom_vjp
def s4d_tap(u, ys, e, K0):
    """replayed S4D layer: value = the taped output ys; cotangent of u = K0 * (cotangent of ys) + e (e = the part from later steps)"""
    return ys


def _tap_fwd(u, ys, e, K0):
    return ys, (e, K0)


def _tap_bwd(res, ct):
    e, K0 = res
    return ct * K0 + e, ct, jnp.zeros_like(e), jnp.zeros_like(K0)


s4d_tap.defvjp(_tap_fwd, _tap_bwd)


def _iota(shape, d):
    return lax.broadcasted_iota(jnp.int32, shape, d)


def step_pre(kp, q, c, k, st):
    """monitor, physics-state rescaling, features, encoder: (encoder output, aux = values the post stage needs)"""
    dt = q["mu"].dtype
    t = k.astype(dt) * M.DT if hasattr(k, "astype") else jnp.asarray(k * M.DT, dt)
    ce = jnp.maximum(c["x3"], 0.0)
    l50 = (q["mu"] + c["ou"] + q["stim"] * (t > q["stim_t"]).astype(dt) + q["stim2"] * (t > q["stim_t2"]).astype(dt)
           + q["stim3"] * (t > q["stim_t3"]).astype(dt))
    cg = jnp.power(ce, q["gamma"])
    b = q["e0"] - q["emax"] * cg / (cg + jnp.power(jnp.exp(l50), q["gamma"]))
    sh = (_iota((DP, DP), 0) + 1 == _iota((DP, DP), 1)).astype(dt)              # buf[:, j] -> buf[:, j + 1]
    first = (_iota((1, DP), 1) == 0).astype(dt)
    shifted = mm(c["buf"], sh) + b[:, None] * first
    k0 = (k == 0)
    buf = jnp.where(k0[:, None] if jnp.ndim(k0) else k0, b[:, None] * jnp.ones((1, DP), dt), shifted)
    y = jnp.sum(buf * q["dmask"], -1) + st["sy"] * st["z0"]
    s = jnp.clip((M.E0 - y) / M.EMAX, 1e-3, 1 - 1e-3)
    w = jnp.minimum(M.C50N * jnp.power(s / (1 - s), 1.0 / M.GAMMA), 3 * M.C50N)
    r, rho, tg = st["r"], st["rho"], st["tg"]
    f0, f1, f3 = (y - tg) / 50, c["up"] / (r * M.UMAX), w / (r * M.C50N) - 1.0
    x = f0[:, None] * kp["enc0"] + f1[:, None] * kp["enc1"] + (0.0 * t)[..., None] * kp["enc2"] + f3[:, None] * kp["enc3"] + kp["enc_b"]
    return x, dict(b=b, buf=buf, w=w, hpr=c["hpr"] * rho[:, None], hpi=c["hpi"] * rho[:, None])


def layer(kp, l, x, ys):
    """residual layer l after its S4D: GELU, gated mixing, residual, layer norm (masked layout: over the active channels, and the layer is
    skipped beyond the network's depth)"""
    lp = kp["layers"][l]
    a = gelu(ys + lp["D"] * x)
    v = x + (mm(a, lp["Wv"]) + lp["bv"]) * jax.nn.sigmoid(mm(a, lp["Wg"]) + lp["bg"])
    if "mH" not in kp:
        m = jnp.mean(v, -1, keepdims=True)
        dv = v - m
        return dv / jnp.sqrt(jnp.mean(dv * dv, -1, keepdims=True) + 1e-5) * lp["g"] + lp["beta"]
    mH, nH, sc = kp["mH"], kp["sc"][4], kp["sc"]
    m = jnp.sum(v * mH, -1, keepdims=True) / nH
    dv = (v - m) * mH
    out = dv / jnp.sqrt(jnp.sum(dv * dv, -1, keepdims=True) / nH + 1e-5) * lp["g"] + lp["beta"]
    return jnp.where(sc[6 + l] > 0, out, x)


def step_post(kp, q, c, st, x, aux):
    """head, action, physics modes, plant, drift, cost: (carry', cost)"""
    sc = kp["sc"]
    r, tg, b, w, hpr, hpi = st["r"], st["tg"], aux["b"], aux["w"], aux["hpr"], aux["hpi"]
    if "decl" in kp:                                                   # masked layout: head or linear decoder per network
        o = jnp.where(sc[5] > 0, jnp.sum(gelu(mm(x, kp["hW"]) + kp["hb"]) * kp["dec"], -1), jnp.sum(x * kp["decl"], -1)) + sc[1]
    else:
        if "hW" in kp:
            x = gelu(mm(x, kp["hW"]) + kp["hb"])
        o = jnp.sum(x * kp["dec"], -1) + sc[1]
    u_lin = sc[0] + jnp.sum(kp["Cor"] * hpr - kp["Coi"] * hpi, -1)
    uu = jnp.clip(r * (u_lin + 10.0 * o), 0.0, M.UMAX)
    un, wn = (uu / r)[:, None], (w / r)[:, None]
    c2 = dict(hpr=kp["lamr"] * hpr - kp["lami"] * hpi + kp["Bur"] * un + kp["Bwr"] * wn,
              hpi=kp["lamr"] * hpi + kp["lami"] * hpr + kp["Bui"] * un + kp["Bwi"] * wn,
              ou=q["a_ou"] * c["ou"] + q["osc"] * st["z1"], up=uu, buf=aux["buf"])
    xs = (c["x0"], c["x1"], c["x2"], c["x3"])
    for i in range(4):
        c2[f"x{i}"] = q[f"A{4 * i}"] * xs[0] + q[f"A{4 * i + 1}"] * xs[1] + q[f"A{4 * i + 2}"] * xs[2] + q[f"A{4 * i + 3}"] * xs[3] + q[f"B{i}"] * uu
    cost = ((b - tg) / 10) ** 2 + sc[2] * (jax.nn.relu(40 - b) / 10) ** 2 + sc[3] * (uu / M.UMAX) ** 2
    return c2, cost


def step(kp, q, c, k, st, s4d):
    """one closed-loop step for a block of patients. kp: kernel params (scalars in kp["sc"]); q: per-patient constants (rows); c: small carry;
    k: step index (scalar, or one per row); st: per-step inputs of this step (rows); s4d(l, u) -> (ys, extra).
    Returns (carry', cost, true BIS, [layer inputs], [extras])."""
    x, aux = step_pre(kp, q, c, k, st)
    us, exs = [], []
    for l in range(len(kp["layers"])):
        us.append(x)
        ys, ex = s4d(l, x)
        exs.append(ex)
        x = layer(kp, l, x, ys)
    c2, cost = step_post(kp, q, c, st, x, aux)
    return c2, cost, aux["b"], us, exs


def s4d_update(lp, hr, hi, u):
    """the S4D recurrence on planes: (ys, h') with h' = dA h + dB u, ys = 2 Re sum C h'"""
    ue = u[..., None]
    h2r = lp["Ar"] * hr - lp["Ai"] * hi + lp["Br"] * ue
    h2i = lp["Ar"] * hi + lp["Ai"] * hr + lp["Bi"] * ue
    return 2.0 * jnp.sum(h2r * lp["Cr"] - h2i * lp["Ci"], -1), (h2r, h2i)


def adjoint_s4d(lp, lr, li, cy):
    """one backward step of the S4D adjoint: incoming lam_h (cotangent of h'), the S4D output cotangent cy -> (e, lam_h for the previous step)"""
    e = jnp.sum(lr * lp["Br"] + li * lp["Bi"], -1)
    gr = lr + 2.0 * cy[..., None] * lp["Cr"]
    gi = li - 2.0 * cy[..., None] * lp["Ci"]
    return e, (gr * lp["Ar"] + gi * lp["Ai"], gi * lp["Ar"] - gr * lp["Ai"])


def replay(kp, q, c, k, st, ys, e):
    """the step with the S4D outputs taken from the tape: (carry', cost) as a function of (kp, carry, ys); also returns the S4D layer inputs"""
    c2, cost, b, us, _ = step(kp, q, c, k, st, lambda l, u: (s4d_tap(u, ys[l], e[l], kp["layers"][l]["K0"]), None))
    return (c2, cost), us


def bwd_step(kp, q, c, k, st, ys, lam_c, lam_h, inv_n):
    """one step of the backward sweep: returns (e per layer, cotangent of the carry before the step, lam_h before the step)"""
    e = [jnp.sum(lr * lp["Br"] + li * lp["Bi"], -1) for lp, (lr, li) in zip(kp["layers"], lam_h)]
    c_bar, cy = bwd_core(kp, q, c, k, st, ys, e, lam_c, inv_n)
    return e, c_bar, [adjoint_s4d(kp["layers"][l], lr, li, cy[l])[1] for l, (lr, li) in enumerate(lam_h)]


def bwd_core(kp, q, c, k, st, ys, e, lam_c, inv_n):
    """the VJP of step k given e: returns (cotangent of the carry before the step, cotangent of every S4D output).
    Taken stage by stage (pre, each layer, post), so a kernel holds only one layer's weights at a time."""
    L = len(kp["layers"])
    (x, aux), pre_vjp = jax.vjp(lambda cc: step_pre(kp, q, cc, k, st), c)
    us = []
    for l in range(L):
        us.append(x)
        x = layer(kp, l, x, ys[l])                           # forward recompute: the tap's value is the taped output
    (c2, cost), post_vjp = jax.vjp(lambda cc, xx, au: step_post(kp, q, cc, st, xx, au), c, x, aux)
    c_bar, cx, caux = post_vjp((lam_c, inv_n * st["w"]))
    cy = [None] * L
    for l in reversed(range(L)):
        def lf(uu, yy, l=l):
            return layer(kp, l, uu, s4d_tap(uu, yy, e[l], kp["layers"][l]["K0"]))
        _, lvjp = jax.vjp(lf, us[l], ys[l])
        cx, cy[l] = lvjp(cx)
    return jax.tree_util.tree_map(jnp.add, c_bar, pre_vjp((cx, caux))[0]), cy


# ---------------------------------------------------------------- reference sweeps (XLA)
def _row(a, k):
    return lax.dynamic_index_in_dim(a, k, 0, keepdims=False)


def block_mats(lp, T):
    """blocked S4D of one layer over T steps (complex): Wb[..., i] = C dA^(i+1), Wu[..., j] = dB dA^(T-1-j) (H, NS, T), dA^T (H, NS), and the
    impulse response K[:, m] = 2 Re sum C dB dA^m (H, T)"""
    dA, dB, C = lp["Ar"] + 1j * lp["Ai"], lp["Br"] + 1j * lp["Bi"], lp["Cr"] + 1j * lp["Ci"]
    pw = [jnp.ones_like(dA)]
    for _ in range(T):
        pw.append(pw[-1] * dA)
    Pw = jnp.stack(pw, -1)
    return dict(Wb=C[..., None] * Pw[..., 1:], Wu=dB[..., None] * Pw[..., T - 1::-1], AT=Pw[..., T], K=2 * jnp.real(jnp.sum((C * dB)[..., None] * Pw[..., :T], 1)))


def _rdot(hr, hi, W):
    """Re sum_n h W for h = hr + i hi (B, H, NS) and complex W (H, NS, T): (B, H, T), one real GEMM"""
    return jnp.einsum("bhn,hnt->bht", jnp.concatenate([hr, hi], -1), jnp.concatenate([jnp.real(W), -jnp.imag(W)], 1), precision=HI)


def _cdot(U, W):
    """sum_t U_t W_t for real U (B, H, T) and complex W (H, NS, T): planes (B, H, NS)"""
    return (jnp.einsum("bht,hnt->bhn", U, jnp.real(W), precision=HI), jnp.einsum("bht,hnt->bhn", U, jnp.imag(W), precision=HI))


def xla_forward(kp, q, st, tape=True, T=0):
    """forward sweep. Returns (tape dict(c = small carry per step (N, B, ...), ys = [(N, B, H)] * L) or None, per-patient sums dict(cost, se, viol)).
    T > 0: blocked S4D (the state is read and written once per T steps instead of every step; same outputs)"""
    if T:
        return _forward_blocked(kp, q, st, tape, T)
    B, Pp = q["mu"].shape[0], kp["lamr"].shape[0]
    H, NS = kp["layers"][0]["Ar"].shape
    h0 = [(jnp.zeros((B, H, NS), q["mu"].dtype),) * 2 for _ in kp["layers"]]

    def body(carry, k):
        c, h = carry
        s = {n: _row(v, k) for n, v in st.items()}
        def s4d(l, u):
            ys, h2 = s4d_update(kp["layers"][l], *h[l], u)
            return ys, (ys, h2)
        c2, cost, b, _, ex = step(kp, q, c, k, s, s4d)
        out = (cost * s["w"], (b - s["tg"]) ** 2, jax.nn.relu(40 - b) ** 2)
        if tape:
            out = out + (c, [y for y, _ in ex])
        return (c2, [hh for _, hh in ex]), out

    _, out = lax.scan(body, (carry0(B, Pp, q["mu"].dtype), h0), jnp.arange(M.N))
    sums = dict(cost=out[0].sum(0), se=out[1].sum(0), viol=out[2].sum(0))
    return (dict(c=out[3], ys=out[4]) if tape else None), sums


def _forward_blocked(kp, q, st, tape, T):
    B, Pp, dt = q["mu"].shape[0], kp["lamr"].shape[0], q["mu"].dtype
    H, NS = kp["layers"][0]["Ar"].shape
    L = len(kp["layers"])
    bm = [block_mats(lp, T) for lp in kp["layers"]]

    def body(carry, kb):
        c, h = carry
        base = [2.0 * _rdot(hr, hi, m["Wb"]) for (hr, hi), m in zip(h, bm)]           # state part of every output of the block
        us = [[] for _ in range(L)]
        outs = []
        for i in range(T):
            k = kb * T + i
            s = {n: _row(v, k) for n, v in st.items()}

            def s4d(l, u, i=i):
                ys = base[l][..., i] + sum(bm[l]["K"][:, i - j] * us[l][j] for j in range(i)) + kp["layers"][l]["K0"] * u
                us[l].append(u)
                return ys, ys
            c2, cost, b, _, ex = step(kp, q, c, k, s, s4d)
            out = (cost * s["w"], (b - s["tg"]) ** 2, jax.nn.relu(40 - b) ** 2)
            outs.append(out + (c, ex) if tape else out)
            c = c2
        h2 = []
        for (hr, hi), m, ul in zip(h, bm, us):
            ur, ui = _cdot(jnp.stack(ul, -1), m["Wu"])
            ATr, ATi = jnp.real(m["AT"]), jnp.imag(m["AT"])
            h2.append((ATr * hr - ATi * hi + ur, ATr * hi + ATi * hr + ui))
        return (c, h2), jax.tree_util.tree_map(lambda *x: jnp.stack(x), *outs)

    h0 = [(jnp.zeros((B, H, NS), dt),) * 2 for _ in range(L)]
    _, out = lax.scan(body, (carry0(B, Pp, dt), h0), jnp.arange(M.N // T))
    out = jax.tree_util.tree_map(lambda a: a.reshape((M.N,) + a.shape[2:]), out)
    sums = dict(cost=out[0].sum(0), se=out[1].sum(0), viol=out[2].sum(0))
    return (dict(c=out[3], ys=out[4]) if tape else None), sums


def _backward_blocked(kp, q, st, tape, T):
    """the backward sweep with the blocked S4D adjoint. With lam = the cotangent of the state at the end of the block (from later blocks):
    e_i = <lam, dB dA^(T-1-i)> + sum_{m>i in the block} K_(m-i) cy_m, and the cotangent of the state at the block start is
    lam conj(dA^T) + 2 sum_m cy_m conj(C dA^(m+1))"""
    B, dt = q["mu"].shape[0], q["mu"].dtype
    L = len(kp["layers"])
    bm = [block_mats(lp, T) for lp in kp["layers"]]
    lam_c = jax.tree_util.tree_map(jnp.zeros_like, carry0(B, kp["lamr"].shape[0], dt))
    lam_h = [(jnp.zeros((B,) + lp["Ar"].shape, dt),) * 2 for lp in kp["layers"]]

    def body(carry, kb):
        lc, lh = carry
        E = [_rdot(lr, -li, m["Wu"]) for (lr, li), m in zip(lh, bm)]                 # <lam, Wu_i> = Re sum lam conj(Wu_i)... as planes
        cys = [[None] * T for _ in range(L)]
        outs = [None] * T
        for i in reversed(range(T)):
            k = kb * T + i
            s = {n: _row(v, k) for n, v in st.items()}
            c = {n: _row(v, k) for n, v in tape["c"].items()}
            ys = [_row(y, k) for y in tape["ys"]]
            e = [E[l][..., i] + sum(bm[l]["K"][:, m - i] * cys[l][m] for m in range(i + 1, T)) for l in range(L)]
            c_bar, cy = bwd_core(kp, q, c, k, s, ys, e, lc, 1.0 / M.N)
            outs[i] = (lc, e)
            for l in range(L):
                cys[l][i] = cy[l]
            lc = c_bar
        lh2 = []
        for (lr, li), m, cl in zip(lh, bm, cys):
            gr, gi = _cdot(2.0 * jnp.stack(cl, -1), m["Wb"])                           # 2 sum cy_m C dA^(m+1)  (conjugated below)
            ATr, ATi = jnp.real(m["AT"]), jnp.imag(m["AT"])
            lh2.append((lr * ATr + li * ATi + gr, li * ATr - lr * ATi - gi))
        return (lc, lh2), jax.tree_util.tree_map(lambda *x: jnp.stack(x), *outs)

    _, (lam, e) = lax.scan(body, (lam_c, lam_h), jnp.arange(M.N // T), reverse=True)
    rs = lambda a: a.reshape((M.N,) + a.shape[2:])
    return dict(lam=jax.tree_util.tree_map(rs, lam), e=[rs(x) for x in e])


def xla_backward(kp, q, st, tape, T=0):
    """backward sweep. Returns dict(lam = cotangent of the carry AFTER each step (N, B, ...), e = [(N, B, H)] * L). T > 0: blocked S4D adjoint"""
    if T:
        return _backward_blocked(kp, q, st, tape, T)
    B = q["mu"].shape[0]
    lam_c = jax.tree_util.tree_map(jnp.zeros_like, carry0(B, kp["lamr"].shape[0], q["mu"].dtype))
    lam_h = [(jnp.zeros((B,) + lp["Ar"].shape, q["mu"].dtype),) * 2 for lp in kp["layers"]]

    def body(carry, k):
        lc, lh = carry
        s = {n: _row(v, k) for n, v in st.items()}
        c = {n: _row(v, k) for n, v in tape["c"].items()}
        ys = [_row(y, k) for y in tape["ys"]]
        e, c_bar, lh2 = bwd_step(kp, q, c, k, s, ys, lc, lh, 1.0 / M.N)
        return (c_bar, lh2), (lc, e)

    _, (lam, e) = lax.scan(body, (lam_c, lam_h), jnp.arange(M.N), reverse=True)
    return dict(lam=lam, e=e)


# ---------------------------------------------------------------- assembly of the per-patient gradients (XLA, parallel over steps)
def s4d_kernel(lp, mS, n=M.N):
    """K_m = 2 Re sum_n C dB dA^m, m = 0..n-1: (H, n)"""
    dt, A = M.s4d_A(lp["log_dt"], lp["log_A_re"], lp["A_im"])
    dtA = dt * A
    dB = (jnp.exp(dtA) - 1) / A * mS
    C = lp["C"][..., 0] + 1j * lp["C"][..., 1]
    pw = jnp.exp(dtA[..., None] * jnp.arange(n))                                  # (H, NS, n)
    return 2 * jnp.real(jnp.sum((C * dB)[..., None] * pw, 1))


def xcorr(cy, u):
    """g[m] = sum_k cy[k] u[k - m] along the last axis (FFT)"""
    n = cy.shape[-1]
    nf = p2(2 * n)
    return jnp.fft.irfft(jnp.fft.rfft(cy, nf) * jnp.conj(jnp.fft.rfft(u, nf)), nf)[..., :n]


def assemble(tp, sz, w_low, rho_u, plain, q, st, ftape, btape, chunk=None):
    """per-patient gradients (tree like tp with a leading batch axis) from the two tapes; the S4D kernels by FFT cross-correlation and the
    kernel's VJP"""
    B = q["mu"].shape[0]
    kp, kp_vjp = jax.vjp(lambda t: to_kp(t, sz, w_low, rho_u, plain), tp)
    mS = P.masks(sz, P.shape_of(tp), tp["phys"]["u0"].dtype)["mS"]
    ks = jnp.arange(M.N)

    def one(qb, stb, cb, ysb, lamb, eb):
        """one patient, all steps as the batch axis"""
        qr = {n: jnp.broadcast_to(v, (M.N,) + v.shape) for n, v in qb.items()}
        f = lambda kp_, ys_: replay(kp_, qr, cb, ks, stb, ys_, eb)
        (_, _), vjp, us = jax.vjp(f, kp, ysb, has_aux=True)
        g_kp, cy = vjp((lamb, stb["w"] / M.N))
        return g_kp, cy, us

    def chunk_fn(i):
        sl = lambda a, ax: lax.dynamic_slice_in_dim(a, i * chunk, chunk, ax)
        args = ({n: sl(v, 0) for n, v in q.items()}, {n: sl(v, 1) for n, v in st.items()}, {n: sl(v, 1) for n, v in ftape["c"].items()},
                [sl(y, 1) for y in ftape["ys"]], {n: sl(v, 1) for n, v in btape["lam"].items()}, [sl(e, 1) for e in btape["e"]])
        axes = (0, 1, 1, 1, 1, 1)
        g_kp, cy, us = jax.vmap(one, in_axes=axes)(*args)
        g = jax.vmap(lambda gk: kp_vjp(gk)[0])(g_kp)
        gl = g["extra"]["layers"]
        for l in range(len(us)):                                                 # S4D kernels: FFT cross-correlation, then the kernel's VJP
            gK = xcorr(jnp.swapaxes(cy[l], 1, 2), jnp.swapaxes(us[l], 1, 2))      # (chunk, H, N)
            _, kvjp = jax.vjp(lambda p: s4d_kernel(p, mS), {k: tp["extra"]["layers"][k][l] for k in ("log_dt", "log_A_re", "A_im", "C")})
            gs = jax.vmap(lambda x: kvjp(x)[0])(gK)
            for k, v in gs.items():
                gl[k] = gl[k].at[:, l].add(v)
        return g

    if chunk is None:                                   # patients per chunk: the S4D-kernel VJP holds chunk x H x NS x N complex values
        d = kp_dims(tp)
        chunk = max(4, min(64, 1 << int(math.log2(max(1, 65536 // (d["H"] * d["NS"]))))))
    chunk = next(c for c in (chunk, 64, 32, 16, 8, 4, 2, 1) if c <= B and B % c == 0)
    out = lax.map(chunk_fn, jnp.arange(B // chunk))
    return jax.tree_util.tree_map(lambda a: a.reshape((B,) + a.shape[2:]), out)


def adjoint_grads(tp, sz, pts, w_low, rho_u, plain=None, forward=xla_forward, backward=xla_backward, chunk=None, assembly=None):
    """per-patient gradients, loss and J of one network with the given sweep and assembly implementations (default: the XLA reference)"""
    kp = to_kp(tp, sz, w_low, rho_u, plain)
    q, st = to_kq(pts)
    ftape, sums = forward(kp, q, st)
    btape = backward(kp, q, st, ftape)
    g = (assembly or assemble)(tp, sz, w_low, rho_u, plain, q, st, ftape, btape, chunk)
    return g, sums["cost"] / M.N, sums["se"] / M.N
