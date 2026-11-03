"""
Propofol closed loop: 3-compartment PK + effect site -> Hill PD -> BIS, with drift, stimulus, monitor delay and noise.

Plant      x' = A(p) x + B(p) u,  x = [c1, c2, c3, ce] (ug/mL), u in [0, 100] mg/min, zero-order hold, dt = 5 s, 30 min (360 steps)
PD         BIS = E0 - Emax * ce^g / (ce^g + C50_t^g),  log C50_t = mu + OU drift (+ a stimulus: a step of random size at a random time)
Monitor    y_k = BIS_{k - delay} + sy_k * noise
Cost       J = mean_k (BIS_k - target_k)^2 on the TRUE BIS; viol = mean relu(40 - BIS)^2

Every patient draws its own values from a scenario spec (ranges; lo == hi is a fixed value), so one scenario mixes many conditions.
The patient population is sampled in JAX (`sample_patients`); the same batch, converted to numpy, drives the numpy plant used for the baselines.
"""
import math, os
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm

DT = 5.0 / 60.0                      # min
T_END = 30.0
N = int(os.environ.get("PKPD_N", round(T_END / DT)))    # steps simulated (360 = the 30 min scenarios; PKPD_N: longer episodes)
TAU_MAX = 30.0                       # min: the slowest S4D mode's time constant (the scale of the shortest training episode)
UMAX, TARGET, GAMMA, E0, EMAX = 100.0, 50.0, 2.0, 100.0, 100.0
PK_KEYS = ("V1", "V2", "V3", "Cl1", "Cl2", "Cl3", "ke0")
NOM = dict(V1=4.27, V2=18.9, V3=238.0, Cl1=1.89, Cl2=1.29, Cl3=0.836, ke0=0.456)     # Schnider, 40 y, 70 kg, 170 cm
C50N = 4.5
PK_CV, C50_CV, SY = 0.2, 0.3, 2.5
DMAX = 12                            # longest monitor delay (steps): 60 s
KMAX = 8                             # most target / noise-level changes per episode
IND = np.arange(N) * DT <= 10.0      # induction phase (first 10 min)

# ---------------------------------------------------------------- scenarios
SPEC_NOMINAL = dict(tau_lo=20.0, tau_hi=20.0, sd_lo=0.15, sd_hi=0.15, stim_p=0.0, stim_lo=0.15, stim_hi=0.45, stim_tlo=5.0, stim_thi=28.0, stim_neg=0.0, stim_nmax=1,
                    dmin=0, dmax=0,
                    gamma_lo=GAMMA, gamma_hi=GAMMA, e0_lo=E0, e0_hi=E0, emax_lo=EMAX, emax_hi=EMAX,
                    tgt_lo=TARGET, tgt_hi=TARGET, pk_shift=0.0, sy_lo=SY, sy_hi=SY,
                    tgt_changes=0, tgt_dmax=10.0, sy_changes=0)


def spec(**kw):
    bad = set(kw) - set(SPEC_NOMINAL)
    assert not bad, f"unknown spec keys {bad}"
    return {**SPEC_NOMINAL, **kw}


# training distribution: everything at once (the delay range is ramped up by train.py)
TRAIN = spec(tau_lo=5.0, tau_hi=40.0, sd_lo=0.05, sd_hi=0.25, stim_p=0.5, dmin=0, dmax=12,
             gamma_lo=1.2, gamma_hi=4.5, e0_lo=90.0, e0_hi=100.0, emax_lo=80.0, emax_hi=100.0,
             tgt_lo=45.0, tgt_hi=60.0, pk_shift=0.4, sy_lo=1.5, sy_hi=5.0, tgt_changes=4, tgt_dmax=10.0, sy_changes=4)

# validation scenarios: name -> (spec, weight in the score)
VAL = {
    "nominal":      (spec(), 2.0),
    "stimulus":     (spec(stim_p=1.0), 1.0),
    "fast_drift":   (spec(tau_lo=5.0, tau_hi=5.0, sd_lo=0.2, sd_hi=0.2), 1.0),
    "delay_30s":    (spec(dmin=6, dmax=6), 1.0),
    "delay_60s":    (spec(dmin=12, dmax=12), 1.0),
    "gamma_wide":   (spec(gamma_lo=1.2, gamma_hi=4.5), 1.0),
    "pk_shift":     (spec(pk_shift=0.5), 1.0),
    "target_range": (spec(tgt_lo=45.0, tgt_hi=60.0), 1.0),
    "target_steps": (spec(tgt_lo=45.0, tgt_hi=60.0, tgt_changes=4, tgt_dmax=10.0), 1.0),
    "noise_steps":  (spec(sy_lo=1.5, sy_hi=6.0, sy_changes=4), 1.0),
    "mixed_hard":   (dict(TRAIN), 2.0),
}

# held-out stress scenarios (beyond the training ranges), evaluation only
STRESS = {
    "gamma_ood_lo":     spec(gamma_lo=1.0, gamma_hi=1.0),
    "gamma_ood_hi":     spec(gamma_lo=6.0, gamma_hi=6.0),
    "pk_shift_ood":     spec(pk_shift=0.9),
    "target_42":        spec(tgt_lo=42.0, tgt_hi=42.0),
    "target_65":        spec(tgt_lo=65.0, tgt_hi=65.0),
    "noise_8":          spec(sy_lo=8.0, sy_hi=8.0),
    "target_steps_ood": spec(tgt_lo=42.0, tgt_hi=65.0, tgt_changes=8, tgt_dmax=20.0),
    "noise_bursts":     spec(sy_lo=2.5, sy_hi=12.0, sy_changes=8),
    "everything_ood":   spec(tau_lo=3.0, tau_hi=60.0, sd_lo=0.05, sd_hi=0.35, stim_p=0.7, dmin=3, dmax=15, gamma_lo=1.0, gamma_hi=6.0,
                             e0_lo=85.0, e0_hi=100.0, emax_lo=70.0, emax_hi=100.0, tgt_lo=42.0, tgt_hi=65.0, pk_shift=0.8,
                             sy_lo=1.5, sy_hi=8.0, tgt_changes=8, tgt_dmax=20.0, sy_changes=8),
    "stimulus_neg":     spec(stim_p=1.0, stim_neg=1.0),          # the stimulus lowers C50 (e.g. stimulation ending): overdose risk
    "stimulus_signed":  spec(stim_p=1.0, stim_neg=0.5, stim_nmax=3),   # 1 to 3 steps, each raising or lowering C50
}
# the stimulus of the meta-optimization (search.py): +0.3 in log C50 after 20 min. A policy can learn to anticipate a fixed time, so the
# selected model is fine-tuned (finetune.py) and evaluated on the random stimulus of the specs above.
STIM_FIXED = dict(stim_lo=0.3, stim_hi=0.3, stim_tlo=20.0, stim_thi=20.0)


def with_stimulus(sp, stimulus):
    """spec `sp` with the "random" (as specified), the "signed" (1 to 3 steps, each of random size, time and sign, half raising and half
    lowering C50: training) or the "fixed" (STIM_FIXED) stimulus"""
    assert stimulus in ("random", "signed", "fixed"), stimulus
    if stimulus == "fixed":
        return {**sp, **STIM_FIXED}
    return {**sp, "stim_neg": 0.5, "stim_nmax": 3} if stimulus == "signed" else dict(sp)


def score_stimulus(stimulus):
    """the stimulus the selection score is taken with: "signed" is a training distribution, the score keeps the standard validation scenarios"""
    return "random" if stimulus == "signed" else stimulus


def val_specs(stimulus="random"):
    """the validation scenarios' specs, name -> spec"""
    return {k: with_stimulus(v[0], stimulus) for k, v in VAL.items()}


DMAX_ALL = max(int(s["dmax"]) for s in [TRAIN, *[v[0] for v in VAL.values()], *STRESS.values()])   # delay buffer length


def score(stats, safe_w=2.0):
    """selection score: weighted mean over the validation scenarios of J + safe_w * viol. stats: name -> dict(J, viol)"""
    w = np.array([VAL[n][1] for n in VAL])
    v = np.array([stats[n]["J"] + safe_w * stats[n]["viol"] for n in VAL])
    return float((w * v).sum() / w.sum())


# ---------------------------------------------------------------- PK / PD
def pk_matrices(p):
    A = np.array([[-(p["Cl1"] + p["Cl2"] + p["Cl3"]) / p["V1"], p["Cl2"] / p["V1"], p["Cl3"] / p["V1"], 0],
                  [p["Cl2"] / p["V2"], -p["Cl2"] / p["V2"], 0, 0],
                  [p["Cl3"] / p["V3"], 0, -p["Cl3"] / p["V3"], 0],
                  [p["ke0"], 0, 0, -p["ke0"]]])
    return A, np.array([[1.0 / p["V1"]], [0], [0], [0]])


def discretize(A, B):
    M = np.zeros((5, 5)); M[:4, :4] = A; M[:4, 4:] = B
    E = expm(M * DT)
    return E[:4, :4], E[:4, 4:]


def pk_disc(p):
    """batched zero-order-hold discretization (JAX). p: (B, 7) in PK_KEYS order -> Ad (B, 4, 4), Bd (B, 4)"""
    V1, V2, V3, Cl1, Cl2, Cl3, ke0 = (p[:, i] for i in range(7))
    B = p.shape[0]
    M = jnp.zeros((B, 5, 5))
    M = M.at[:, 0, 0].set(-(Cl1 + Cl2 + Cl3) / V1).at[:, 0, 1].set(Cl2 / V1).at[:, 0, 2].set(Cl3 / V1)
    M = M.at[:, 1, 0].set(Cl2 / V2).at[:, 1, 1].set(-Cl2 / V2)
    M = M.at[:, 2, 0].set(Cl3 / V3).at[:, 2, 2].set(-Cl3 / V3)
    M = M.at[:, 3, 0].set(ke0).at[:, 3, 3].set(-ke0)
    M = M.at[:, 0, 4].set(1.0 / V1)
    E = jax.vmap(jax.scipy.linalg.expm)(M * DT)
    return E[:, :4, :4], E[:, :4, 4]


def s4d_A(log_dt, log_A_re, A_im):
    """continuous S4D eigenvalues with the decay per step floored at DT / TAU_MAX (no mode slower than TAU_MAX): (dt (..., 1), A)"""
    dt = jnp.exp(log_dt)[..., None]
    rate = jnp.maximum(dt * jnp.exp(log_A_re), DT / TAU_MAX)
    return dt, -rate / dt + 1j * A_im


def hill(ce, logc50, gamma, e0, emax, xp=jnp):
    ce = xp.maximum(ce, 0.0)
    return e0 - emax * ce ** gamma / (ce ** gamma + xp.exp(logc50) ** gamma)


def cref(c50, gamma, e0, emax, target):
    """effect-site concentration that gives BIS = target"""
    s = (e0 - target) / emax
    return c50 * (s / (1 - s)) ** (1.0 / gamma)


# ---------------------------------------------------------------- patient sampling (JAX)
def sample_patients(key, B, sp, nom_frac=0.0):
    """B patients from spec `sp` (values may be traced scalars, e.g. the ramped dmax). A share `nom_frac` of them is exactly the nominal case
    (nominal drift, no stimulus / delay / PD or PK shift, target 50, noise SY); only their PK / C50 inter-individual variability remains.
    Returns a dict of arrays, all batch-first, plus the noise of the whole episode (N, 2, B): [measurement, drift]."""
    f = lambda v: jnp.asarray(v, jnp.float32)
    lens = "len_hi" in sp                                                # per-patient episode lengths: only training specs carry len_lo / len_hi (min)
    sp = {k: f(v) for k, v in sp.items()}
    ks = jax.random.split(key, 16)
    u = lambda k, shape=(B,): jax.random.uniform(k, shape)
    lu = lambda k, lo, hi: jnp.exp(jnp.log(lo) + u(k) * (jnp.log(hi) - jnp.log(lo)))
    un = lambda k, lo, hi: lo + u(k) * (hi - lo)
    nom = u(ks[0]) < nom_frac

    logpk = PK_CV * jax.random.normal(ks[1], (B, 7))
    shift = jnp.where(nom, 0.0, sp["pk_shift"] * u(ks[2]))               # common shift of volumes / clearances / ke0
    zc = jax.random.normal(ks[3], (B, 3))
    logpk = logpk + shift[:, None] * zc[:, jnp.array([0, 0, 0, 1, 1, 1, 2])]
    pk = jnp.asarray([NOM[k] for k in PK_KEYS], jnp.float32) * jnp.exp(logpk)
    Ad, Bd = pk_disc(pk)

    pick = lambda v, v0: jnp.where(nom, v0, v)
    steps = jnp.arange(N)

    def traj(k, init, n_max, new_level):
        """(B, N) piecewise-constant trajectory: 0..n_max changes at random steps, each to a new random level"""
        k1, k2, k3 = jax.random.split(k, 3)
        n = jnp.where(nom, 0.0, jnp.floor(u(k1) * (n_max + 1.0)))
        times = jnp.sort(jnp.floor(u(k2, (B, KMAX)) * N), axis=1)
        lv, prev = [init], init
        for kj in jax.random.split(k3, KMAX):
            prev = new_level(prev, u(kj))
            lv.append(prev)
        active = jnp.arange(KMAX)[None, :] < n[:, None]
        cnt = jnp.sum(active[:, :, None] & (times[:, :, None] <= steps[None, None, :]), 1)
        return jnp.take_along_axis(jnp.stack(lv, 1), cnt, axis=1)

    tgt0 = pick(un(ks[4], sp["tgt_lo"], sp["tgt_hi"]), TARGET)
    tgt = traj(ks[5], tgt0, sp["tgt_changes"],
               lambda prev, v: jnp.clip(prev + (2 * v - 1) * sp["tgt_dmax"], sp["tgt_lo"], sp["tgt_hi"]))
    sy0 = pick(un(ks[6], sp["sy_lo"], sp["sy_hi"]), SY)
    sy = traj(ks[7], sy0, sp["sy_changes"], lambda prev, v: jnp.exp(jnp.log(sp["sy_lo"]) + v * (jnp.log(sp["sy_hi"]) - jnp.log(sp["sy_lo"]))))
    nomv = jnp.broadcast_to(nom[:, None], tgt.shape)
    tgt, sy = jnp.where(nomv, TARGET, tgt), jnp.where(nomv, SY, sy)
    delay = jnp.floor(sp["dmin"] + u(ks[8]) * (sp["dmax"] - sp["dmin"] + 1)).astype(jnp.int32)
    kst = jax.random.split(jax.random.fold_in(key, 2), 2)                  # stimulus: whether (ks[12]), how large (log C50 step), when (min)
    stim = jnp.where(u(ks[12]) < sp["stim_p"], un(kst[0], sp["stim_lo"], sp["stim_hi"]), 0.0)
    neg = jax.random.uniform(jax.random.fold_in(key, 3), (B,)) < sp["stim_neg"]     # its own stream: stim_neg = 0 leaves every draw unchanged
    stim = pick(jnp.where(neg, -stim, stim), 0.0)
    # further steps (stim_nmax > 1): a stimulated patient has 1 .. stim_nmax steps, each with its own size, sign and time (own streams)
    nst = 1.0 + jnp.floor(jax.random.uniform(jax.random.fold_in(key, 4), (B,)) * sp["stim_nmax"])
    more = {}
    for j in (2, 3):
        kj = jax.random.split(jax.random.fold_in(key, 3 + 2 * j), 3)
        aj = un(kj[0], sp["stim_lo"], sp["stim_hi"])
        aj = jnp.where(u(kj[1]) < sp["stim_neg"], -aj, aj)
        more[f"stim{j}"] = jnp.where((stim != 0) & (nst >= j), aj, 0.0)
        more[f"stim_t{j}"] = un(kj[2], sp["stim_tlo"], sp["stim_thi"])
    return dict(
        pk=pk, Ad=Ad, Bd=Bd,
        mu=math.log(C50N) + C50_CV * jax.random.normal(ks[9], (B,)),
        tau=pick(lu(ks[10], sp["tau_lo"], sp["tau_hi"]), 20.0),
        sd=pick(un(ks[11], sp["sd_lo"], sp["sd_hi"]), 0.15),
        stim=stim, stim_t=un(kst[1], sp["stim_tlo"], sp["stim_thi"]), **more,
        **({"len": jnp.clip(jnp.floor(un(jax.random.fold_in(key, 11), sp["len_lo"], sp["len_hi"]) / DT), 1, N).astype(jnp.int32)} if lens else {}),
        delay=jnp.where(nom, 0, delay),
        gamma=pick(lu(ks[13], sp["gamma_lo"], sp["gamma_hi"]), GAMMA),
        e0=pick(un(ks[14], sp["e0_lo"], sp["e0_hi"]), E0),
        emax=pick(un(ks[15], sp["emax_lo"], sp["emax_hi"]), EMAX),
        tgt=tgt, sy=sy,
        z=jax.random.normal(jax.random.fold_in(key, 1), (N, 2, B)))


def to_numpy(pts):
    return {k: np.asarray(v, np.float64) if k != "delay" else np.asarray(v) for k, v in pts.items()}


def take(pts, idx):
    """sub-batch of a numpy patient batch (the noise is time-major)"""
    return {k: (v[:, :, idx] if k == "z" else v[idx]) for k, v in pts.items()}


# ---------------------------------------------------------------- numpy plant (vectorized over patients), for the baselines
class Plant:
    """numpy reference of the plant for a patient batch (dict from sample_patients, converted with to_numpy).
    Usage: y = plant.measure(); u = controller(...); plant.step(u)."""

    def __init__(self, pts):
        self.p = pts
        self.B = pts["mu"].shape[0]
        self.x = np.zeros((self.B, 4))
        self.ou = np.zeros(self.B)
        self.k = 0
        self.hist = np.zeros((N, self.B))
        self.us = np.zeros((N, self.B))

    def logc50(self, k=None):
        k = self.k if k is None else k
        p = self.p
        return (p["mu"] + self.ou + p["stim"] * (k * DT > p["stim_t"]) + p.get("stim2", 0.0) * (k * DT > p.get("stim_t2", 0.0))
                + p.get("stim3", 0.0) * (k * DT > p.get("stim_t3", 0.0)))

    def bis(self):
        return hill(self.x[:, 3], self.logc50(), self.p["gamma"], self.p["e0"], self.p["emax"], xp=np)

    def measure(self):
        k = self.k
        self.hist[k] = self.bis()
        bd = self.hist[np.maximum(k - self.p["delay"], 0), np.arange(self.B)]
        return bd + self.p["sy"][:, k] * self.p["z"][k, 0]

    def step(self, u):
        k = self.k
        self.us[k] = u
        self.x = np.einsum("bij,bj->bi", self.p["Ad"], self.x) + self.p["Bd"] * u[:, None]
        a = np.exp(-DT / self.p["tau"])
        self.ou = a * self.ou + self.p["sd"] * np.sqrt(1 - a * a) * self.p["z"][k, 1]
        self.k += 1


def run_numpy(controller, pts):
    """closed loop of a numpy controller (reset(pts, plant), act(k, y) -> u (B,)) on a numpy patient batch -> (BIS (B, N), u (B, N))"""
    plant = Plant(pts)
    controller.reset(pts, plant)
    for k in range(N):
        y = plant.measure()
        u = np.clip(np.asarray(controller.act(k, y), np.float64), 0.0, UMAX)
        plant.step(u)
    return plant.hist.T, plant.us.T


def metrics(b, u, tgt):
    """per-patient metrics of (B, N) traces"""
    se = (b - tgt) ** 2
    late = ~IND
    reach = b <= tgt + 5.0
    return dict(J=se.mean(1), J_ind=se[:, IND].mean(1), Jm=se[:, late].mean(1), low=100.0 * (b < 40).mean(1),
                viol=(np.maximum(40 - b, 0.0) ** 2).mean(1), drug=u.sum(1) * DT,
                t_ind=np.where(reach.any(1), reach.argmax(1) * DT, np.nan),
                band=100.0 * (np.abs(b - tgt) <= 10.0)[:, late].mean(1))
