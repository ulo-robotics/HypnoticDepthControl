"""
The data of the XLA engine, all NamedTuples (pytrees) in the natural layout: complex S4D and physics states, the plant as a 4x4 matrix, the
layers stacked along a leading axis.

  Net       the network of one member in its own (possibly padded) shape: the physics compensator's modes, the encoder, the discretized S4D
            layers (Layers, stacked), the head, the decoders and the masks of a padded network (active channels, head switch, layer gates)
  Patients  per-patient constants of the closed loop (B, ...)
  Steps     per-step inputs (N, B): noise, target, measurement-noise level, reference scalings, the cost weight of the step
  Carry     the small state carried from step to step (plant, drift, last infusion, physics modes, delay line)
  Tape      the forward sweep's tape: the carry before every step and the S4D outputs of every layer
  Adjoint   the backward sweep's output: the cotangent of the carry after every step and the S4D-recurrence part of every S4D input's cotangent
  Sums      per-patient sums of the forward sweep: the weighted cost, the squared error, the violation below BIS 40
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

from lethe.core import model as M
from lethe.core import policy as P

DP = M.DMAX_ALL + 1                  # delay line length
HI = lax.Precision.HIGHEST           # exact float32 matmuls (no TF32)


class Layers(NamedTuple):
    """S4D residual layers, stacked (L, ...); one layer when indexed with layer_at"""
    dA: jax.Array        # (L, H, NS) complex: decay per step
    dB: jax.Array        # (L, H, NS) complex: input map (zero on the padded states)
    C: jax.Array         # (L, H, NS) complex: readout
    K0: jax.Array        # (L, H): zero-lag tap 2 Re sum C dB
    D: jax.Array         # (L, H): skip
    Wv: jax.Array        # (L, H, H): gated mixing, value half
    Wg: jax.Array        # (L, H, H): gated mixing, gate half
    bv: jax.Array        # (L, H)
    bg: jax.Array        # (L, H)
    g: jax.Array         # (L, H): layer norm gain
    beta: jax.Array      # (L, H): layer norm bias


class Net(NamedTuple):
    lam: jax.Array       # (P,) complex: physics modes
    Bu: jax.Array        # (P,) complex: their infusion input
    Bw: jax.Array        # (P,) complex: their effect-site input
    Cout: jax.Array      # (P,) complex: their readout
    u0: jax.Array        # (): physics offset
    dec_b: jax.Array     # (): decoder bias
    w_low: jax.Array     # (): loss weight below BIS 40
    rho_u: jax.Array     # (): loss weight of the infusion
    enc_W: jax.Array     # (3, H): encoder
    enc_b: jax.Array     # (H,)
    layers: Layers
    hW: jax.Array        # (H, HD): head
    hb: jax.Array        # (HD,)
    hdec: jax.Array      # (HD,): decoder after the head
    dec_lin: jax.Array   # (H,): linear decoder (no head)
    mH: jax.Array        # (H,): active channels
    nH: jax.Array        # (): their count
    head: jax.Array      # (): 1 with the head, 0 with the linear decoder
    gate: jax.Array      # (L,): 1 for the network's own layers, 0 beyond its depth


class Patients(NamedTuple):
    Ad: jax.Array        # (B, 4, 4): discretized PK
    Bd: jax.Array        # (B, 4)
    mu: jax.Array        # (B,): log C50
    a_ou: jax.Array      # (B,): drift decay per step
    osc: jax.Array       # (B,): drift innovation scale
    gamma: jax.Array     # (B,): Hill slope
    e0: jax.Array        # (B,)
    emax: jax.Array      # (B,)
    stim: jax.Array      # (B,): stimulation steps in log C50 and their times (min)
    stim_t: jax.Array
    stim2: jax.Array
    stim_t2: jax.Array
    stim3: jax.Array
    stim_t3: jax.Array
    dmask: jax.Array     # (B, DP): one-hot monitor delay


class Steps(NamedTuple):
    z0: jax.Array        # (N, B): measurement noise
    z1: jax.Array        # (N, B): drift noise
    tg: jax.Array        # (N, B): target
    sy: jax.Array        # (N, B): measurement-noise level
    r: jax.Array         # (N, B): reference effect site / C50N
    rho: jax.Array       # (N, B): reference ratio previous / now (rescales the physics modes)
    w: jax.Array         # (N, B): cost weight of the step (N [k < len] / len; 1 for full-length episodes)


class Carry(NamedTuple):
    x: jax.Array         # (B, 4): plant
    ou: jax.Array        # (B,): drift of log C50
    up: jax.Array        # (B,): previous infusion
    hp: jax.Array        # (B, P) complex: physics modes
    buf: jax.Array       # (B, DP): BIS j steps ago


class Tape(NamedTuple):
    c: Carry             # (N, B, ...): the carry before every step
    ys: jax.Array        # (L, N, B, H): S4D outputs


class Adjoint(NamedTuple):
    lam: Carry           # (N, B, ...): cotangent of the carry after every step (JAX's convention for the complex modes)
    e: jax.Array         # (L, N, B, H): the S4D-recurrence part of the cotangent of every S4D input


class Sums(NamedTuple):
    cost: jax.Array      # (B,): sum over the steps of the weighted training cost
    se: jax.Array        # (B,): sum of (BIS - target)^2
    viol: jax.Array      # (B,): sum of relu(40 - BIS)^2


def layer_at(layers, l):
    """layer l of stacked Layers"""
    return Layers(*[a[l] for a in layers])


def nsteps(steps):
    return steps.tg.shape[0]


def complex_of(dtype):
    return jnp.result_type(dtype, jnp.complex64)


def net_of(tp, sz, w_low, rho_u):
    """training-layout parameters (core/policy.py) of a network of sizes sz -> Net (differentiable; the discretization of policy.prep)"""
    dt = tp["phys"]["u0"].dtype
    mk = P.masks(sz, P.shape_of(tp), dt)
    (lam, Bu, Bw, Cout), (dA, dB, C) = P.prep(tp, mk)
    ex, ly = tp["extra"], tp["extra"]["layers"]
    layers = Layers(dA=dA, dB=dB, C=C, K0=2 * jnp.sum(jnp.real(C * dB), -1), D=ly["D"], Wv=ly["Wv"], Wg=ly["Wg"], bv=ly["bv"], bg=ly["bg"],
                    g=ly["ln_g"], beta=ly["ln_b"])
    return Net(lam=lam, Bu=Bu, Bw=Bw, Cout=Cout, u0=tp["phys"]["u0"][0], dec_b=ex["dec_b"][0], w_low=jnp.asarray(w_low, dt),
               rho_u=jnp.asarray(rho_u, dt), enc_W=ex["enc_W"], enc_b=ex["enc_b"], layers=layers, hW=ex["hW"], hb=ex["hb"],
               hdec=ex["hdec"][:, 0], dec_lin=ex["dec_lin"][:, 0], mH=mk["mH"], nH=mk["nH"], head=mk["head"].astype(dt),
               gate=mk["gate"].astype(dt))


def patients_of(pts):
    """patient batch (model.sample_patients) -> Patients"""
    a = jnp.exp(-M.DT / pts["tau"])
    return Patients(Ad=pts["Ad"], Bd=pts["Bd"], mu=pts["mu"], a_ou=a, osc=pts["sd"] * jnp.sqrt(1 - a * a), gamma=pts["gamma"], e0=pts["e0"],
                    emax=pts["emax"], stim=pts["stim"], stim_t=pts["stim_t"], stim2=pts["stim2"], stim_t2=pts["stim_t2"], stim3=pts["stim3"],
                    stim_t3=pts["stim_t3"], dmask=jax.nn.one_hot(pts["delay"], DP, dtype=pts["mu"].dtype))


def steps_of(pts):
    """patient batch -> Steps; a batch with per-patient lengths (pts["len"]) weights each patient's steps by N [k < len] / len"""
    tg = pts["tgt"].T
    tgp = jnp.concatenate([tg[:1], tg[:-1]], 0)
    if "len" in pts:
        N = tg.shape[0]
        on = jnp.arange(N)[:, None] < pts["len"][None, :]
        w = jnp.where(on, N / pts["len"][None, :].astype(tg.dtype), 0.0).astype(tg.dtype)
    else:
        w = jnp.ones_like(tg)
    return Steps(z0=pts["z"][:, 0], z1=pts["z"][:, 1], tg=tg, sy=pts["sy"].T, r=P.w_ref(tg) / M.C50N, rho=P.w_ref(tgp) / P.w_ref(tg), w=w)


def carry0(B, P_, dtype):
    """the carry at the start of an episode (B patients, P_ physics modes)"""
    z = jnp.zeros((B,), dtype)
    return Carry(x=jnp.zeros((B, 4), dtype), ou=z, up=z, hp=jnp.zeros((B, P_), complex_of(dtype)), buf=jnp.full((B, DP), M.E0, dtype))


def zeros_like_carry(c):
    return Carry(*[jnp.zeros_like(a) for a in c])


def blocks(steps, T):
    """Steps (N, B) -> (N / T, T, B): the blocks of T steps of a blocked sweep"""
    N = nsteps(steps)
    return Steps(*[a.reshape((N // T, T) + a.shape[1:]) for a in steps])
