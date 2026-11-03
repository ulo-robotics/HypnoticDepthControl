"""
The policy: a physics-initialized S4D controller, u = r * (u_lin + 10 * residual).

u_lin     the "physics layer": a steady-state Kalman observer (PK + a random-walk C50 state) + LQR with a moving set-point, designed on the
          nominal patient and written in modal coordinates. Its 5 complex modes (a, b), input / measurement gains (Bu, Bw) and output (Cout)
          are trainable. Input is w = the effect-site concentration that the nominal Hill model maps to the measured BIS.
residual  encoder (4 features -> H) -> L S4D layers (diagonal complex SSM, NS states per channel, gated mixing, layer norm) -> optional
          MLP head -> scalar. The decoder starts at zero, so the untrained policy IS the prior compensator.
r         w_ref(target) / C50N: the physics branch runs in target-normalized units, so a different target BIS rescales it (r = 1 at 50).

Two parameter layouts:
  compact   (files, deployment, C++ export) a plain dict of arrays, the layout of the old best_model.pkl files:
              phys  = dict(a, b, Bu, Bw, Cout, u0)          (P,) (P,) (P,2) (P,2) (P,2) (1,)   complex values stored as [..., (re, im)], P = 5 + n_extra
              extra = dict(enc_W (3,H), enc_b (H,), layers=[...], dec_W (H or head,1), dec_b (1,), [head_W1 (H,head), head_b1 (head,)])
              layer = dict(log_dt (H,), log_A_re (H,NS), A_im (H,NS), C (H,NS,2), D (H,), W_mix (H,2H), b_mix (2H,), ln_g (H,), ln_b (H,))
  training  the same values with one tree structure for every size: the layers stacked along a leading axis, the gated-mixing halves as
            separate matrices (Wv, Wg, bv, bg), and the head always present (hW, hb, hdec; dec_lin is the decoder of a head-less network).
            A network of sizes sz = dict(H, NS, L, head) can be zero-padded to any larger shape (H, NS, L, HD, P): masks computed from sz switch
            the padding off (S4D inputs of padded states, layer-norm statistics of padded channels, layers beyond L, the unused decoder), the
            padded entries get exactly zero gradients and so stay zero under every optimizer. A padded network computes and trains like the
            compact one; search.py trains a population of different sizes as one program this way.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from scipy.linalg import block_diag, solve_discrete_are

import model as M

# hyperparameters of the architecture and of the prior design (the design defines the initial policy)
ARCH_DEFAULTS = dict(H=64, NS=8, L=3, head=64, n_extra=0, q_theta=2.5e-3, sy_scale=0.78, rho_lqr=8.6e-4, q_x=1.2e-4)


# ---------------------------------------------------------------- prior design (numpy / scipy)
def design_prior(q_theta, sy_scale, rho_lqr, q_x):
    """steady-state Kalman observer on [c1, c2, c3, ce, C50 offset] + LQR on ce, closed observer loop in modal coordinates"""
    Ad, Bd = M.discretize(*M.pk_matrices(M.NOM))
    A5, B5 = block_diag(Ad, np.eye(1)), np.vstack([Bd, [[0.0]]])
    C5 = np.array([[0, 0, 0, 1.0, 1.0]])
    slope = M.EMAX * M.GAMMA / (4 * M.C50N)                    # |dBIS / dce| at C50: noise in concentration units
    Rw = np.array([[(M.SY * sy_scale / slope) ** 2]])
    P = solve_discrete_are(A5.T, C5.T, np.diag([q_x] * 4 + [M.C50N ** 2 * q_theta]), Rw)
    Lg = A5 @ P @ C5.T @ np.linalg.inv(C5 @ P @ C5.T + Rw)
    Qc = np.zeros((4, 4)); Qc[3, 3] = 1.0
    R = np.array([[rho_lqr]])
    Pc = solve_discrete_are(Ad, Bd, Qc, R)
    K = np.linalg.solve(R + Bd.T @ Pc @ Bd, Bd.T @ Pc @ Ad)
    Kfull = np.hstack([K, [[M.NOM["Cl1"] + K.sum()]]])
    lam, V = np.linalg.eig(A5 - Lg @ C5)
    Vinv = np.linalg.inv(V)
    return dict(lam=lam, Bu=(Vinv @ B5)[:, 0], Bw=(Vinv @ Lg)[:, 0], Cout=-(Kfull @ V)[0], u0=float((M.NOM["Cl1"] + K.sum()) * M.C50N))


def init_params(key, cfg):
    """untrained policy for the config (architecture sizes + prior design), compact layout; its output equals the prior compensator.
    n_extra > 0 adds physics modes with time constants 2..400 min, random input gains and zero output (so the output is unchanged)."""
    H, NS, L, hd = (int(cfg[k]) for k in ("H", "NS", "L", "head"))
    ne = int(cfg.get("n_extra", 0))
    d = design_prior(*(float(cfg[k]) for k in ("q_theta", "sy_scale", "rho_lqr", "q_x")))
    logl = np.log(d["lam"].astype(complex))
    ri = lambda z: np.stack([np.real(z), np.imag(z)], -1)
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    a, b, Bu, Bw, Cout = np.log(-np.real(logl)), np.imag(logl), ri(d["Bu"]), ri(d["Bw"]), ri(d["Cout"])
    if ne > 0:
        ke = jax.random.split(jax.random.fold_in(key, 99), 2)
        a = np.concatenate([a, np.log(1.0 / np.geomspace(2.0, 400.0, ne))])
        b = np.concatenate([b, np.zeros(ne)])
        Bu = np.concatenate([Bu, 0.1 * np.asarray(jax.random.normal(ke[0], (ne, 2)))])
        Bw = np.concatenate([Bw, 0.1 * np.asarray(jax.random.normal(ke[1], (ne, 2)))])
        Cout = np.concatenate([Cout, np.zeros((ne, 2))])
    phys = dict(a=f32(a), b=f32(b), Bu=f32(Bu), Bw=f32(Bw), Cout=f32(Cout), u0=f32([d["u0"]]))
    ks = jax.random.split(key, 2 + 4 * L)
    nrm = lambda k, s, sc: sc * jax.random.normal(k, s)
    layers = []
    for i in range(L):
        k1, k2, k3, k4 = ks[2 + 4 * i: 6 + 4 * i]
        layers.append(dict(
            log_dt=jax.random.uniform(k1, (H,)) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3),
            log_A_re=jnp.full((H, NS), math.log(0.5)), A_im=math.pi * jnp.tile(jnp.arange(NS, dtype=jnp.float32), (H, 1)),
            C=nrm(k3, (H, NS, 2), NS ** -0.5), D=nrm(k4, (H,), 1.0),
            W_mix=nrm(k2, (H, 2 * H), H ** -0.5), b_mix=jnp.zeros(2 * H), ln_g=jnp.ones(H), ln_b=jnp.zeros(H)))
    extra = dict(enc_W=nrm(ks[0], (3, H), 0.5), enc_b=jnp.zeros(H), layers=layers, dec_b=jnp.zeros(1))
    if hd > 0:
        extra.update(head_W1=nrm(ks[1], (H, hd), H ** -0.5), head_b1=jnp.zeros(hd), dec_W=jnp.zeros((hd, 1)))
    else:
        extra["dec_W"] = jnp.zeros((H, 1))
    return dict(phys=phys, extra=extra)


def sizes(params):
    """(H, NS, L, head) of a compact parameter tree"""
    ex = params["extra"]
    H, NS = ex["layers"][0]["log_A_re"].shape
    return H, NS, len(ex["layers"]), (ex["head_b1"].shape[0] if "head_W1" in ex else 0)


# ---------------------------------------------------------------- training layout
_DIMS = {                       # leaf name -> its dims in the training layout (a size name or a fixed length)
    "a": ("P",), "b": ("P",), "Bu": ("P", 2), "Bw": ("P", 2), "Cout": ("P", 2), "u0": (1,),
    "enc_W": (3, "H"), "enc_b": ("H",), "dec_b": (1,), "dec_lin": ("H", 1), "hW": ("H", "HD"), "hb": ("HD",), "hdec": ("HD", 1),
    "log_dt": ("L", "H"), "log_A_re": ("L", "H", "NS"), "A_im": ("L", "H", "NS"), "C": ("L", "H", "NS", 2), "D": ("L", "H"),
    "Wv": ("L", "H", "H"), "Wg": ("L", "H", "H"), "bv": ("L", "H"), "bg": ("L", "H"), "ln_g": ("L", "H"), "ln_b": ("L", "H")}


def size_of(cfg):
    return dict(H=int(cfg["H"]), NS=int(cfg["NS"]), L=int(cfg["L"]), head=int(cfg["head"]))


def shape_of(tp):
    """dict(H, NS, L, HD, P) of a training-layout tree (leading population axes are ignored)"""
    ex, ly = tp["extra"], tp["extra"]["layers"]
    return dict(H=ex["enc_b"].shape[-1], NS=ly["log_A_re"].shape[-1], L=ly["D"].shape[-2], HD=ex["hb"].shape[-1], P=tp["phys"]["a"].shape[-1])


def expand(cp):
    """compact -> training layout at its own sizes. Returns (tree, sz)"""
    ex, (H, NS, L, hd) = cp["extra"], sizes(cp)
    st = lambda f: jnp.stack([f(lp) for lp in ex["layers"]])
    layers = {k: st(lambda lp, k=k: lp[k]) for k in ("log_dt", "log_A_re", "A_im", "C", "D", "ln_g", "ln_b")}
    layers.update(Wv=st(lambda lp: lp["W_mix"][:, :H]), Wg=st(lambda lp: lp["W_mix"][:, H:]), bv=st(lambda lp: lp["b_mix"][:H]),
                  bg=st(lambda lp: lp["b_mix"][H:]))
    z = lambda *s: jnp.zeros(s, ex["enc_b"].dtype)
    if hd:
        head = dict(dec_lin=z(H, 1), hW=ex["head_W1"], hb=ex["head_b1"], hdec=ex["dec_W"])
    else:
        head = dict(dec_lin=ex["dec_W"], hW=z(H, 1), hb=z(1), hdec=z(1, 1))
    extra = dict(enc_W=ex["enc_W"], enc_b=ex["enc_b"], dec_b=ex["dec_b"], layers=layers, **head)
    return dict(phys=dict(cp["phys"]), extra=extra), dict(H=H, NS=NS, L=L, head=hd)


def shrink(tp, cfg):
    """training layout (possibly padded) -> compact layout of the config's sizes (numpy)"""
    tp = jax.tree_util.tree_map(np.asarray, tp)
    H, NS, L, hd = (int(cfg[k]) for k in ("H", "NS", "L", "head"))
    n = 5 + int(cfg.get("n_extra", 0))
    ph, ex, ly = tp["phys"], tp["extra"], tp["extra"]["layers"]
    phys = {k: (v if k == "u0" else v[:n]) for k, v in ph.items()}
    layers = [dict(log_dt=ly["log_dt"][l, :H], log_A_re=ly["log_A_re"][l, :H, :NS], A_im=ly["A_im"][l, :H, :NS], C=ly["C"][l, :H, :NS],
                   D=ly["D"][l, :H], W_mix=np.concatenate([ly["Wv"][l, :H, :H], ly["Wg"][l, :H, :H]], 1),
                   b_mix=np.concatenate([ly["bv"][l, :H], ly["bg"][l, :H]]), ln_g=ly["ln_g"][l, :H], ln_b=ly["ln_b"][l, :H]) for l in range(L)]
    extra = dict(enc_W=ex["enc_W"][:, :H], enc_b=ex["enc_b"][:H], layers=layers, dec_b=ex["dec_b"])
    if hd:
        extra.update(head_W1=ex["hW"][:H, :hd], head_b1=ex["hb"][:hd], dec_W=ex["hdec"][:hd])
    else:
        extra["dec_W"] = ex["dec_lin"][:H]
    return dict(phys=phys, extra=extra)


def resize(tree, shape):
    """zero-pad / crop every parameter-shaped leaf of a tree (parameters, EMA, optimizer state) to a training-layout shape (numpy)"""
    def f(path, a):
        name = next((k.key for k in reversed(path) if isinstance(k, jax.tree_util.DictKey)), None)
        dims = _DIMS.get(name)
        if dims is None or np.ndim(a) != len(dims):
            return a
        a = np.asarray(a)
        out = np.zeros(tuple(shape[d] if isinstance(d, str) else d for d in dims), a.dtype)
        sl = tuple(slice(0, min(x, y)) for x, y in zip(a.shape, out.shape))
        out[sl] = a[sl]
        return out
    return jax.tree_util.tree_map_with_path(f, tree)


def masks(sz, shape, dtype=jnp.float32):
    """masks of a network of sizes sz inside a padded shape"""
    return dict(mS=(jnp.arange(shape["NS"]) < sz["NS"]).astype(dtype), mH=(jnp.arange(shape["H"]) < sz["H"]).astype(dtype),
                nH=jnp.asarray(sz["H"], dtype), gate=jnp.arange(shape["L"]) < sz["L"], head=jnp.asarray(sz["head"]) > 0)


def training_form(params, sz=None):
    """(training-layout tree, sz) from either layout"""
    return expand(params) if isinstance(params["extra"]["layers"], list) else (params, sz)


# ---------------------------------------------------------------- the policy (JAX, batched over patients; training layout)
def prep(tp, mk):
    """per-step constants of the recurrences (discretized SSMs); computed once per rollout"""
    ph, ly = tp["phys"], tp["extra"]["layers"]
    cx = lambda m: m[..., 0] + 1j * m[..., 1]
    phys = (jnp.exp(-jnp.exp(ph["a"]) + 1j * ph["b"]), cx(ph["Bu"]), cx(ph["Bw"]), cx(ph["Cout"]))
    dt, A = M.s4d_A(ly["log_dt"], ly["log_A_re"], ly["A_im"])          # time constants bounded by TAU_MAX
    dA = jnp.exp(dt * A)
    return phys, (dA, (dA - 1) / A * mk["mS"], cx(ly["C"]))


def init_state(tp, B):
    """(physics modes (B, P), S4D states [(B, H, NS)] * L), complex"""
    s = shape_of(tp)
    ct = jnp.result_type(tp["phys"]["a"].dtype, jnp.complex64)
    return (jnp.zeros((B, s["P"]), ct), [jnp.zeros((B, s["H"], s["NS"]), ct) for _ in range(s["L"])])


def w_from_y(y):
    """effect-site concentration that the nominal Hill model maps to BIS y"""
    s = jnp.clip((M.E0 - y) / M.EMAX, 1e-3, 1 - 1e-3)
    return jnp.minimum(M.C50N * (s / (1 - s)) ** (1.0 / M.GAMMA), 3 * M.C50N)


def w_ref(target):
    return M.cref(M.C50N, M.GAMMA, M.E0, M.EMAX, target)


def s4d_layer(lp, pre, x, h, mk):
    """one layer: S4D, GELU, gated mixing, residual, layer norm over the active channels"""
    dA, dB, C = pre
    h = dA * h + dB * x[..., None]
    a = jax.nn.gelu(2 * jnp.real(jnp.sum(h * C, -1)) + lp["D"] * x)
    v = x + (a @ lp["Wv"] + lp["bv"]) * jax.nn.sigmoid(a @ lp["Wg"] + lp["bg"])
    m = jnp.sum(v * mk["mH"], -1, keepdims=True) / mk["nH"]
    dv = (v - m) * mk["mH"]
    return dv / jnp.sqrt(jnp.sum(dv * dv, -1, keepdims=True) / mk["nH"] + 1e-5) * lp["ln_g"] + lp["ln_b"], h


def act(tp, mk, pre, state, y, u_prev, t, tgt, tgt_prev):
    """one control step for a batch: measured BIS y, previous infusion u_prev, time t (min), reference now / at the previous step.
    Returns (infusion in [0, UMAX], new state)."""
    (lam, Bu, Bw, Cout), (dA, dB, C) = pre
    hp, hs = state
    r = w_ref(tgt) / M.C50N
    hp = hp * (w_ref(tgt_prev) / w_ref(tgt))[:, None]               # re-express the physics state when the reference moves
    w = w_from_y(y)
    ex = tp["extra"]
    feat = jnp.stack([(y - tgt) / 50, u_prev / (r * M.UMAX), w / (r * M.C50N) - 1.0], -1)
    x = feat @ ex["enc_W"] + ex["enc_b"]
    new = []
    for l, h in enumerate(hs):
        x2, h = s4d_layer({k: v[l] for k, v in ex["layers"].items()}, (dA[l], dB[l], C[l]), x, h, mk)
        x = jnp.where(mk["gate"][l], x2, x)
        new.append(h)
    o = jnp.where(mk["head"], jax.nn.gelu(x @ ex["hW"] + ex["hb"]) @ ex["hdec"], x @ ex["dec_lin"])[..., 0] + ex["dec_b"][0]
    u_lin = tp["phys"]["u0"][0] + jnp.real(jnp.sum(Cout * hp, -1))
    u = jnp.clip(r * (u_lin + 10.0 * o), 0.0, M.UMAX)
    hp = lam * hp + Bu * (u / r)[:, None] + Bw * (w / r)[:, None]
    return u, (hp, new)


# ---------------------------------------------------------------- closed loop in JAX (training and evaluation)
PATIENT_AXES = {k: (2 if k == "z" else 0) for k in ("pk", "Ad", "Bd", "mu", "tau", "sd", "stim", "stim_t", "stim2", "stim_t2", "stim3", "stim_t3", "delay", "gamma", "e0", "emax", "tgt", "sy", "z")}  # batch axis of each field


def patient_axes(pts):
    """the batch axis of every field a patient batch carries (the noise is time-major; optional fields such as len included)"""
    return {k: (2 if k == "z" else 0) for k in pts}


def rollout(params, pts, w_low=0.0, rho_u=0.0, chunk=20, sz=None):
    """closed loop of the policy (either layout; a training-layout tree needs its sizes sz) on a patient batch (sample_patients).
    Returns (training loss per patient (B,), true BIS (N, B), infusion (N, B)).
    loss = mean_k ((BIS - target) / 10)^2 + w_low (relu(40 - BIS) / 10)^2 + rho_u (u / UMAX)^2. Rematerialized per chunk of steps (memory)."""
    tp, sz = training_form(params, sz)
    mk = masks(sz, shape_of(tp), tp["phys"]["a"].dtype)
    B = pts["mu"].shape[0]
    pre = prep(tp, mk)
    dmask = jax.nn.one_hot(pts["delay"], M.DMAX_ALL + 1, dtype=jnp.float32)

    def step(carry, inp):
        x, ou, state, u_prev, buf = carry
        k, z = inp
        t = k * M.DT
        l50 = (pts["mu"] + ou + pts["stim"] * (t > pts["stim_t"]) + pts["stim2"] * (t > pts["stim_t2"]) + pts["stim3"] * (t > pts["stim_t3"]))
        b = M.hill(x[:, 3], l50, pts["gamma"], pts["e0"], pts["emax"])
        buf = jnp.where(k == 0, b[:, None], jnp.concatenate([b[:, None], buf[:, :-1]], 1))   # buf[:, j] = BIS j steps ago (first reading before that)
        y = jnp.sum(buf * dmask, -1) + pts["sy"][:, k] * z[0]
        tg, tgp = pts["tgt"][:, k], pts["tgt"][:, jnp.maximum(k - 1, 0)]
        u, state = act(tp, mk, pre, state, y, u_prev, t, tg, tgp)
        cost = ((b - tg) / 10) ** 2 + w_low * (jax.nn.relu(40 - b) / 10) ** 2 + rho_u * (u / M.UMAX) ** 2
        a = jnp.exp(-M.DT / pts["tau"])
        x = jnp.einsum("bij,bj->bi", pts["Ad"], x) + pts["Bd"] * u[:, None]
        ou = a * ou + pts["sd"] * jnp.sqrt(1 - a * a) * z[1]
        return (x, ou, state, u, buf), (cost, b, u)

    @jax.checkpoint
    def chunk_fn(carry, inp):
        return lax.scan(step, carry, inp)

    carry = (jnp.zeros((B, 4)), jnp.zeros(B), init_state(tp, B), jnp.zeros(B), jnp.full((B, M.DMAX_ALL + 1), M.E0))
    N = pts["z"].shape[0]
    ks = jnp.arange(N).reshape(-1, chunk)
    zs = pts["z"].reshape(N // chunk, chunk, 2, B)
    _, (cost, b, u) = lax.scan(chunk_fn, carry, (ks, zs))
    cost, b, u = (a.reshape(N, B) for a in (cost, b, u))
    if "len" in pts:                                                       # variable-length episodes: each patient's own steps
        on = jnp.arange(N)[:, None] < pts["len"][None, :]
        return jnp.sum(jnp.where(on, cost, 0.0), 0) / pts["len"], b, u
    return cost.mean(0), b, u


def stats(b, tgt):
    """J and viol over a batch (JAX). b: (N, B), tgt: (B, N)"""
    return dict(J=jnp.mean((b - tgt.T) ** 2), viol=jnp.mean(jax.nn.relu(40 - b) ** 2))


# ---------------------------------------------------------------- numpy-side controller wrapper (evaluation next to the baselines)
class PolicyController:
    """runs the JAX policy step by step on the numpy plant (model.run_numpy); used by the selftest to check rollout() against the reference
    plant, and to record the C++ engine's reference episodes"""

    def __init__(self, params):
        self.tp, sz = training_form(jax.tree_util.tree_map(jnp.asarray, params))
        self.mk = masks(sz, shape_of(self.tp), self.tp["phys"]["a"].dtype)
        self._act = jax.jit(act)

    def reset(self, pts, plant):
        B = pts["mu"].shape[0]
        self.dt = self.tp["phys"]["a"].dtype
        self.tgt = jnp.asarray(pts["tgt"], self.dt)
        self.state = init_state(self.tp, B)
        self.pre = prep(self.tp, self.mk)
        self.u_prev = jnp.zeros(B, self.dt)

    def act(self, k, y):
        u, self.state = self._act(self.tp, self.mk, self.pre, self.state, jnp.asarray(y, self.dt), self.u_prev, jnp.asarray(k * M.DT, self.dt),
                                  self.tgt[:, k], self.tgt[:, max(k - 1, 0)])
        self.u_prev = u
        return np.asarray(u, np.float64)


def load_model(path):
    """a saved model (the best_model.pkl dict: params in the compact layout, cfg, ...). A model saved with the earlier 4-input encoder, whose
    third input (the time t / T_END) was held at zero, has that row dropped: the same policy. One that used the time input is refused."""
    import pickle
    m = pickle.load(open(path, "rb"))
    ex = m["params"]["extra"]
    if ex["enc_W"].shape[0] == 4:
        if (m.get("semantics") or {}).get("time_feature", False):
            raise SystemExit(f"{path}: trained with the time input, which this code no longer has")
        ex["enc_W"] = ex["enc_W"][[0, 1, 3]]
        m.pop("semantics", None)
    return m

