"""
The per-patient gradients from the two tapes. With both tapes every step is independent: per patient, one VJP of the replayed step over all
its steps at once (the steps as the rows of the batch) gives the cotangents of the network (Net) and of every S4D output. The S4D kernels'
parameters (dt, A, C) get theirs through the convolution the S4D layer is (ys = sum_m K_m u_(k-m), K_m = 2 Re sum C dB dA^m): an FFT
cross-correlation of each layer's output cotangent with its input gives the cotangent of K, and K's Jacobian maps it to the parameters. Each
channel's K depends only on that channel's parameters, so the Jacobian has 1 + 4 NS directions (log_dt, then log_A_re, A_im, Re C, Im C per
state), all channels at once: forward-mode derivatives, computed once per assembly; per patient the gradient is then one contraction over
the lags. The Net cotangent goes back to
the training-layout parameters through the VJP of net_of. Patients are processed in chunks (a scan over chunk offsets; the memory of the
kernel VJP grows with the chunk).
"""
import math
import jax
import jax.numpy as jnp
from jax import lax

from lethe.core import model as M
from lethe.core import policy as P
from lethe.engine.xla.layout import HI, Carry, Patients, Steps, layer_at, net_of, nsteps
from lethe.engine.xla.step import layer_tapped, monitor, post

S4D_KEYS = ("log_dt", "log_A_re", "A_im", "C")


def replay(net, pat, c, k, s, ys, e):
    """the step with the S4D outputs from the tape, rows = (patient, step) pairs: ((carry', cost), S4D inputs (L, rows, H))"""
    x, aux = monitor(net, pat, c, k, s)
    us = []
    for l in range(ys.shape[0]):
        us.append(x)
        x = layer_tapped(layer_at(net.layers, l), net.mH, net.nH, net.gate[l], x, ys[l], e[l])
    return post(net, pat, c, s, x, aux), jnp.stack(us)


def patient_cotangents(net, pat, s, c, ys, lam, e):
    """one patient (its Steps, carry tape, S4D outputs, carry cotangents, e: the steps first): (Net cotangent, S4D output cotangents (L, N, H),
    S4D inputs (L, N, H))"""
    N = s.tg.shape[0]
    rows = Patients(*[jnp.broadcast_to(a, (N,) + a.shape) for a in pat])
    _, vjp, us = jax.vjp(replay, net, rows, c, jnp.arange(N), s, ys, e, has_aux=True)
    g = vjp((lam, s.w / N))
    return g[0], g[5], us


def s4d_kernel(lp, mS, lags):
    """K_m = 2 Re sum_n C dB dA^m at the lags m (n,): (L, H, n); lp: the S4D parameters of the training layout"""
    dt, A = M.s4d_A(lp["log_dt"], lp["log_A_re"], lp["A_im"])
    dtA = dt * A
    dB = (jnp.exp(dtA) - 1) / A * mS
    C = lp["C"][..., 0] + 1j * lp["C"][..., 1]
    return 2 * jnp.real(jnp.sum((C * dB)[..., None] * jnp.exp(dtA[..., None] * lags), -2))


def kernel_directions(lp):
    """the 1 + 4 NS tangent directions of the S4D parameters, every channel at once: dict of (D, ...) arrays"""
    L, H, NS = lp["log_A_re"].shape
    D, dt = 1 + 4 * NS, lp["log_A_re"].dtype
    one = jnp.broadcast_to(jnp.eye(NS, dtype=dt)[:, None, None, :], (NS, L, H, NS))
    t_ar = jnp.zeros((D, L, H, NS), dt).at[1:1 + NS].set(one)
    t_ai = jnp.zeros((D, L, H, NS), dt).at[1 + NS:1 + 2 * NS].set(one)
    t_C = jnp.zeros((D, L, H, NS, 2), dt).at[1 + 2 * NS:1 + 3 * NS, ..., 0].set(one).at[1 + 3 * NS:, ..., 1].set(one)
    return dict(log_dt=jnp.zeros((D, L, H), dt).at[0].set(1.0), log_A_re=t_ar, A_im=t_ai, C=t_C)


def kernel_jvp(lp, mS, lags, t):
    """the derivative of the kernel along the parameter direction t: (L, H, n)"""
    return jax.jvp(s4d_kernel, (lp, mS, lags), (t, jnp.zeros_like(mS), jnp.zeros_like(lags)))[1]


def kernel_jacobian(lp, mS, N):
    """dK / d(direction) for every direction of kernel_directions: (D, L, H, N)"""
    lags = jnp.arange(N, dtype=lp["log_A_re"].dtype)
    return jax.vmap(kernel_jvp, in_axes=(None, None, None, 0))(lp, mS, lags, kernel_directions(lp))


def kernel_cotangents(J, gK):
    """per-patient cotangents of the S4D parameters from their kernels' cotangents gK (n, L, H, N) and the Jacobian J (D, L, H, N)"""
    NS = (J.shape[0] - 1) // 4
    gd = jnp.einsum("plhm,dlhm->pdlh", gK, J, precision=HI)
    return dict(log_dt=gd[:, 0], log_A_re=_states(gd, 0, NS), A_im=_states(gd, 1, NS), C=jnp.stack([_states(gd, 2, NS), _states(gd, 3, NS)], -1))


def _states(gd, i, NS):
    """the i-th group of NS directions of gd (n, D, L, H) -> (n, L, H, NS)"""
    return jnp.moveaxis(gd[:, 1 + i * NS:1 + (i + 1) * NS], 1, -1)


def net_cotangent(tp, sz, w_low, rho_u, g_net):
    """the cotangent of the training-layout parameters from a Net cotangent"""
    _, vjp = jax.vjp(net_of, tp, sz, w_low, rho_u)
    return vjp(g_net)[0]


def xcorr(cy, u):
    """g[m] = sum_k cy[k] u[k - m] along the last axis (FFT)"""
    n = cy.shape[-1]
    nf = 1 << (2 * n - 1).bit_length()
    return jnp.fft.irfft(jnp.fft.rfft(cy, nf) * jnp.conj(jnp.fft.rfft(u, nf)), nf)[..., :n]


def _slice(a, start, n, axis):
    return lax.dynamic_slice_in_dim(a, start, n, axis)


def _chunk_body(carry, start):
    """one chunk of patients (offsets start + offsets): their per-patient gradients"""
    tp, sz, w_low, rho_u, net, J, pat, steps, tape, adj, offsets = carry
    n = offsets.shape[0]
    pat_c = Patients(*[_slice(a, start, n, 0) for a in pat])
    steps_c = Steps(*[_slice(a, start, n, 1) for a in steps])
    c_c = Carry(*[_slice(a, start, n, 1) for a in tape.c])
    lam_c = Carry(*[_slice(a, start, n, 1) for a in adj.lam])
    ys_c, e_c = _slice(tape.ys, start, n, 2), _slice(adj.e, start, n, 2)
    g_net, cy, us = jax.vmap(patient_cotangents, in_axes=(None, 0, 1, 1, 2, 1, 2))(net, pat_c, steps_c, c_c, ys_c, lam_c, e_c)
    g = jax.vmap(net_cotangent, in_axes=(None, None, None, None, 0))(tp, sz, w_low, rho_u, g_net)
    gK = xcorr(jnp.swapaxes(cy, -1, -2), jnp.swapaxes(us, -1, -2))                     # (n, L, H, N)
    g_s4 = kernel_cotangents(J, gK)
    layers = dict(g["extra"]["layers"])
    for k in S4D_KEYS:
        layers[k] = layers[k] + g_s4[k]
    return carry, dict(g, extra=dict(g["extra"], layers=layers))


def default_chunk(tp, B):
    """patients per chunk (the replayed steps' VJP holds chunk x N rows of the step)"""
    s = P.shape_of(tp)
    c = max(4, min(64, 1 << int(math.log2(max(1, 65536 // (s["H"] * s["NS"]))))))
    return next(x for x in (c, 64, 32, 16, 8, 4, 2, 1) if x <= B and B % x == 0)


def _merge(a):
    """(chunks, chunk, ...) -> (B, ...)"""
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


def _assemble(tp, sz, w_low, rho_u, pat, steps, tape, adj, chunk=None):
    """per-patient gradients (a tree like tp with a leading batch axis) from the forward tape and the backward sweep's Adjoint"""
    B = pat.mu.shape[0]
    chunk = chunk or default_chunk(tp, B)
    net = net_of(tp, sz, w_low, rho_u)
    mS = P.masks(sz, P.shape_of(tp), tp["phys"]["u0"].dtype)["mS"]
    J = kernel_jacobian({k: tp["extra"]["layers"][k] for k in S4D_KEYS}, mS, nsteps(steps))
    carry = (tp, sz, w_low, rho_u, net, J, pat, steps, tape, adj, jnp.arange(chunk))
    _, g = lax.scan(_chunk_body, carry, jnp.arange(0, B, chunk))
    return jax.tree_util.tree_map(_merge, g)


assemble = jax.jit(_assemble, static_argnames=("chunk",))
