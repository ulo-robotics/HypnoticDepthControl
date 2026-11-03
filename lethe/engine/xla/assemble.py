"""
The per-patient gradients from the two tapes. With both tapes every step is independent: per patient, one VJP of the replayed step over all
its steps at once (the steps as the rows of the batch) gives the cotangents of the network (Net) and of every S4D output. The S4D kernels'
parameters (dt, A, C) get theirs through the convolution the S4D layer is (ys = sum_m K_m u_(k-m), K_m = 2 Re sum C dB dA^m): an FFT
cross-correlation of each layer's output cotangent with its input gives the cotangent of K, then the VJP of K. The Net cotangent goes back to
the training-layout parameters through the VJP of net_of. Patients are processed in chunks (a scan over chunk offsets; the memory of the
kernel VJP grows with the chunk).
"""
import math
import jax
import jax.numpy as jnp
from jax import lax

from lethe.core import model as M
from lethe.core import policy as P
from lethe.engine.xla.layout import Carry, Patients, Steps, layer_at, net_of
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


def kernel_cotangent(lp, mS, gK):
    """the cotangent of the S4D parameters lp from the cotangent gK (L, H, N) of their kernel"""
    lags = jnp.arange(gK.shape[-1], dtype=gK.dtype)
    _, vjp = jax.vjp(s4d_kernel, lp, mS, lags)
    return vjp(gK)[0]


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
    tp, sz, w_low, rho_u, net, mS, pat, steps, tape, adj, offsets = carry
    n = offsets.shape[0]
    pat_c = Patients(*[_slice(a, start, n, 0) for a in pat])
    steps_c = Steps(*[_slice(a, start, n, 1) for a in steps])
    c_c = Carry(*[_slice(a, start, n, 1) for a in tape.c])
    lam_c = Carry(*[_slice(a, start, n, 1) for a in adj.lam])
    ys_c, e_c = _slice(tape.ys, start, n, 2), _slice(adj.e, start, n, 2)
    g_net, cy, us = jax.vmap(patient_cotangents, in_axes=(None, 0, 1, 1, 2, 1, 2))(net, pat_c, steps_c, c_c, ys_c, lam_c, e_c)
    g = jax.vmap(net_cotangent, in_axes=(None, None, None, None, 0))(tp, sz, w_low, rho_u, g_net)
    gK = xcorr(jnp.swapaxes(cy, -1, -2), jnp.swapaxes(us, -1, -2))                     # (n, L, H, N)
    lp = {k: tp["extra"]["layers"][k] for k in S4D_KEYS}
    g_s4 = jax.vmap(kernel_cotangent, in_axes=(None, None, 0))(lp, mS, gK)
    layers = dict(g["extra"]["layers"])
    for k in S4D_KEYS:
        layers[k] = layers[k] + g_s4[k]
    return carry, dict(g, extra=dict(g["extra"], layers=layers))


def default_chunk(tp, B):
    """patients per chunk: the S4D-kernel VJP holds chunk x L x H x NS x N complex values"""
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
    carry = (tp, sz, w_low, rho_u, net, mS, pat, steps, tape, adj, jnp.arange(chunk))
    _, g = lax.scan(_chunk_body, carry, jnp.arange(0, B, chunk))
    return jax.tree_util.tree_map(_merge, g)


assemble = jax.jit(_assemble, static_argnames=("chunk",))
