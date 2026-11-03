"""
The XLA engine: per-patient gradients of the closed-loop training cost by the adjoint method, exact up to rounding, in plain JAX (the
reference the CUDA engine is checked against).

  layout.py    the data (Net, Patients, Steps, Carry, Tape, Adjoint, Sums) and their construction from the training-layout parameters and a
               patient batch
  step.py      the closed-loop step in stages (monitor, layer, post) and the replayed S4D layer (s4d_tap)
  sweeps.py    forward and backward sweeps, per step or with the S4D blocked over T steps
  assembly.py  the per-patient gradients from the two tapes

Every function is a module-level function and every public one is jitted (static: tape, T, chunk); scan bodies get their constants through
the carry. adjoint_grads is the whole gradient of one network. The *_k functions are the engine's interface (lethe/engine): the same sweeps
and assembly on the CUDA kernels' tape format, so that a forward and a backward sweep of different engines can be combined.
"""
import jax
import jax.numpy as jnp

from lethe.engine.xla.assembly import assemble
from lethe.engine.xla.layout import (DP, HI, Adjoint, Carry, Layers, Net, Patients, Steps, Sums, Tape, complex_of, net_of, nsteps,
                                     patients_of, steps_of)
from lethe.engine.xla.sweeps import backward, forward

__all__ = ["DP", "HI", "Adjoint", "Carry", "Layers", "Net", "Patients", "Steps", "Sums", "Tape", "adjoint_grads", "assemble", "backward",
           "forward", "net_of", "nsteps", "patients_of", "steps_of", "forward_k", "backward_k", "assemble_k"]


def _adjoint_grads(tp, sz, pts, w_low, rho_u, T=0, chunk=None):
    """per-patient gradients (tree like tp, leading batch axis), training loss and J (B,) of one network on a patient batch"""
    net, pat, steps = net_of(tp, sz, w_low, rho_u), patients_of(pts), steps_of(pts)
    tape, sums = forward(net, pat, steps, True, T)
    adj = backward(net, pat, steps, tape, T)
    N = nsteps(steps)
    return assemble(tp, sz, w_low, rho_u, pat, steps, tape, adj, chunk), sums.cost / N, sums.se / N


adjoint_grads = jax.jit(_adjoint_grads, static_argnames=("T", "chunk"))


# ---------------------------------------------------------------- the CUDA kernels' tape format
KEYS = ("x0", "x1", "x2", "x3", "ou", "up", "hpr", "hpi", "buf")       # the small carry of the kernels, one array each


def padded_modes(P_):
    """the kernels' physics-mode count: P_ zero-padded to a power of two, at least 8"""
    return max(8, 1 << max(0, P_ - 1).bit_length())


def _pad(a, n):
    return jnp.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])])


def carry_to_kernel(c, cotangent=False):
    """Carry (natural) -> the kernels' dict. The kernels hold the physics modes as planes (re, im); a cotangent there is (dL/dre, dL/dim),
    which is the conjugate of JAX's complex cotangent"""
    Pp = padded_modes(c.hp.shape[-1])
    im = -jnp.imag(c.hp) if cotangent else jnp.imag(c.hp)
    return dict(x0=c.x[..., 0], x1=c.x[..., 1], x2=c.x[..., 2], x3=c.x[..., 3], ou=c.ou, up=c.up, hpr=_pad(jnp.real(c.hp), Pp),
                hpi=_pad(im, Pp), buf=c.buf)


def carry_from_kernel(d, P_, cotangent=False):
    """the kernels' dict -> Carry with P_ physics modes"""
    im = -d["hpi"][..., :P_] if cotangent else d["hpi"][..., :P_]
    return Carry(x=jnp.stack([d["x0"], d["x1"], d["x2"], d["x3"]], -1), ou=d["ou"], up=d["up"],
                 hp=(d["hpr"][..., :P_] + 1j * im).astype(complex_of(d["ou"].dtype)), buf=d["buf"])


def tape_to_kernel(tape):
    return dict(c=carry_to_kernel(tape.c), ys=[tape.ys[l] for l in range(tape.ys.shape[0])])


def tape_from_kernel(ft, P_):
    return Tape(c=carry_from_kernel(ft["c"], P_), ys=jnp.stack(ft["ys"]))


def adjoint_to_kernel(adj):
    return dict(lam=carry_to_kernel(adj.lam, cotangent=True), e=[adj.e[l] for l in range(adj.e.shape[0])])


def adjoint_from_kernel(bt, P_):
    return Adjoint(lam=carry_from_kernel(bt["lam"], P_, cotangent=True), e=jnp.stack(bt["e"]))


def _sums_dict(s):
    return dict(cost=s.cost, se=s.se, viol=s.viol)


def _modes(tp):
    return tp["phys"]["a"].shape[-1]


def forward_k(tp, sz, w_low, rho_u, plain, pts, tape=True, T=0):
    """the engine's forward interface: (the kernels' tape dict or None, sums dict). plain: unused (a layout of the CUDA kernels)"""
    tape_, sums = forward(net_of(tp, sz, w_low, rho_u), patients_of(pts), steps_of(pts), tape, T)
    return (tape_to_kernel(tape_) if tape else None), _sums_dict(sums)


def backward_k(tp, sz, w_low, rho_u, plain, pts, ftape, T=0):
    """the engine's backward interface on the kernels' tape: the kernels' dict(lam, e)"""
    adj = backward(net_of(tp, sz, w_low, rho_u), patients_of(pts), steps_of(pts), tape_from_kernel(ftape, _modes(tp)), T)
    return adjoint_to_kernel(adj)


def assemble_k(tp, sz, w_low, rho_u, plain, pts, ftape, btape, chunk=None):
    """the engine's assembly interface on the kernels' tapes: per-patient gradients"""
    P_ = _modes(tp)
    return assemble(tp, sz, w_low, rho_u, patients_of(pts), steps_of(pts), tape_from_kernel(ftape, P_), adjoint_from_kernel(btape, P_), chunk)
