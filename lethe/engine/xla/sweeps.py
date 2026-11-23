"""
The forward and backward sweeps (lax.scan over the steps; every body is a module-level function and gets its constants through the carry).

forward    runs the episode; with tape=True it also writes the carry before every step and every layer's S4D output. The S4D states never
           go to the tape.
backward   k = N-1 .. 0: replays step k from the tape (S4D outputs taken from it) and takes its VJP stage by stage, carrying the cotangent of
           the small carry and of every S4D state. Writes per step the incoming carry cotangent and e, the S4D-recurrence part of the
           cotangent of every S4D input. No parameter gradient is accumulated here (assemble.py does that from both tapes).

T > 0 blocks the S4D over T steps: one contraction gives the state part of a block's outputs (Wb = C dA^(i+1)), the outputs within the
block add the impulse response of the block's earlier inputs (a strictly lower Toeplitz matrix of K_m = 2 Re sum C dB dA^m), and the state
is read and written once per block. Same results as T = 0 up to rounding. The steps within a block are an inner scan, unrolled (its
length T is static); unroll > 1 also unrolls the outer scan over the steps (or blocks) by that factor.

S4D adjoint (per step, h' = dA h + dB u, ys = 2 Re sum C h'): with lam the cotangent of h' (as lam = dL/dRe h' + i dL/dIm h'),
e = Re sum conj(lam) dB, and the cotangent of h is (lam + 2 cy conj(C)) conj(dA), cy the cotangent of ys.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

from lethe.engine.xla.layout import HI, Adjoint, Carry, Sums, Tape, blocks, carry0, complex_of, layer_at, nsteps, zeros_like_carry
from lethe.engine.xla.step import layer, layer_tapped, monitor, post, s4d_out, sums_row


class BlockMats(NamedTuple):
    """blocked S4D of every layer over T steps"""
    Wb: jax.Array        # (L, H, NS, T) complex: C dA^(i+1), the state part of the block's outputs
    Wu: jax.Array        # (L, H, NS, T) complex: dB dA^(T-1-j), the block's inputs' part of the state at its end
    AT: jax.Array        # (L, H, NS) complex: dA^T
    Kt: jax.Array        # (L, H, T, T): Kt[..., i, j] = K_(i-j) for j < i, else 0


def block_mats(layers, T):
    dA, dB, C = layers.dA, layers.dB, layers.C
    pw = jnp.concatenate([jnp.ones_like(dA)[..., None], jnp.cumprod(jnp.broadcast_to(dA[..., None], dA.shape + (T,)), -1)], -1)  # dA^0 .. dA^T
    K = 2 * jnp.real(jnp.sum((C * dB)[..., None] * pw[..., :T], -2))                                                          # (L, H, T)
    lag = jnp.arange(T)[:, None] - jnp.arange(T)[None, :]
    Kt = jnp.where(lag > 0, K[..., jnp.clip(lag, 0, T - 1)], 0.0)
    return BlockMats(Wb=C[..., None] * pw[..., 1:], Wu=dB[..., None] * pw[..., T - 1::-1], AT=pw[..., T], Kt=Kt)


# ---------------------------------------------------------------- forward, per step
def _fwd_step(net, pat, c, h, k, s):
    """one step with the S4D recurrence: (carry', S4D states', cost, true BIS, S4D outputs (L, B, H))"""
    x, aux = monitor(net, pat, c, k, s)
    hs, yss = [], []
    for l in range(h.shape[0]):
        lp = layer_at(net.layers, l)
        hl = lp.dA * h[l] + lp.dB * x[..., None]
        ys = s4d_out(lp, hl)
        hs.append(hl)
        yss.append(ys)
        x = layer(lp, net.mH, net.nH, net.gate[l], x, ys)
    c2, cost = post(net, pat, c, s, x, aux)
    return c2, jnp.stack(hs), cost, aux.b, jnp.stack(yss)


def _fwd_body(carry, xs):
    net, pat, c, h = carry
    k, s = xs
    c2, h2, cost, b, _ = _fwd_step(net, pat, c, h, k, s)
    return (net, pat, c2, h2), sums_row(cost, b, s)


def _fwd_body_tape(carry, xs):
    net, pat, c, h = carry
    k, s = xs
    c2, h2, cost, b, ys = _fwd_step(net, pat, c, h, k, s)
    return (net, pat, c2, h2), (sums_row(cost, b, s), c, ys)


# ---------------------------------------------------------------- forward, blocked
def _blk_step(net, pat, bm, base, ubuf, c, i, k, s):
    """step i of a block: the S4D outputs from the block's state part and its earlier inputs"""
    x, aux = monitor(net, pat, c, k, s)
    ub, yss = [], []
    for l in range(base.shape[0]):
        lp = layer_at(net.layers, l)
        ys = (lax.dynamic_index_in_dim(base[l], i, -1, keepdims=False)
              + jnp.einsum("bht,ht->bh", ubuf[l], lax.dynamic_index_in_dim(bm.Kt[l], i, 1, keepdims=False), precision=HI) + lp.K0 * x)
        ub.append(lax.dynamic_update_index_in_dim(ubuf[l], x, i, -1))
        yss.append(ys)
        x = layer(lp, net.mH, net.nH, net.gate[l], x, ys)
    c2, cost = post(net, pat, c, s, x, aux)
    return c2, jnp.stack(ub), cost, aux.b, jnp.stack(yss)


def _blk_inner(carry, xs):
    net, pat, bm, base, c, ubuf = carry
    i, k, s = xs
    c2, ubuf2, cost, b, _ = _blk_step(net, pat, bm, base, ubuf, c, i, k, s)
    return (net, pat, bm, base, c2, ubuf2), sums_row(cost, b, s)


def _blk_inner_tape(carry, xs):
    net, pat, bm, base, c, ubuf = carry
    i, k, s = xs
    c2, ubuf2, cost, b, ys = _blk_step(net, pat, bm, base, ubuf, c, i, k, s)
    return (net, pat, bm, base, c2, ubuf2), (sums_row(cost, b, s), c, ys)


def _blk_block(carry, xs, inner):
    """one block: its outputs' state part, the inner scan over its steps, the state at its end"""
    net, pat, bm, c, h = carry
    ks, sb = xs
    T = ks.shape[0]
    base = 2.0 * jnp.real(jnp.einsum("lbhn,lhnt->lbht", h, bm.Wb, precision=HI))
    ubuf = jnp.zeros(base.shape, base.dtype)
    (_, _, _, _, c2, ubuf), out = lax.scan(inner, (net, pat, bm, base, c, ubuf), (jnp.arange(T), ks, sb), unroll=True)
    h2 = bm.AT[:, None] * h + jnp.einsum("lbht,lhnt->lbhn", ubuf.astype(h.dtype), bm.Wu, precision=HI)
    return (net, pat, bm, c2, h2), out


def _blk_body(carry, xs):
    return _blk_block(carry, xs, _blk_inner)


def _blk_body_tape(carry, xs):
    return _blk_block(carry, xs, _blk_inner_tape)


def _flat(a):
    """(N / T, T, ...) -> (N, ...)"""
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


# ---------------------------------------------------------------- forward
def _forward(net, pat, steps, tape=True, T=0, unroll=1):
    """forward sweep of one network: (Tape or None, Sums). tape: write the tape; T > 0: blocked S4D"""
    B, N = pat.mu.shape[0], nsteps(steps)
    L, H, NS = net.layers.dA.shape
    dt = pat.mu.dtype
    c0, h0 = carry0(B, net.lam.shape[0], dt), jnp.zeros((L, B, H, NS), complex_of(dt))
    ks = jnp.arange(N)
    if T:
        bm = block_mats(net.layers, T)
        _, out = lax.scan(_blk_body_tape if tape else _blk_body, (net, pat, bm, c0, h0), (ks.reshape(N // T, T), blocks(steps, T)), unroll=unroll)
        out = jax.tree_util.tree_map(_flat, out)
    else:
        _, out = lax.scan(_fwd_body_tape if tape else _fwd_body, (net, pat, c0, h0), (ks, steps), unroll=unroll)
    rows = out[0] if tape else out
    sums = Sums(cost=rows[0].sum(0), se=rows[1].sum(0), viol=rows[2].sum(0))
    return (Tape(c=out[1], ys=jnp.moveaxis(out[2], 1, 0)) if tape else None), sums


forward = jax.jit(_forward, static_argnames=("tape", "T", "unroll"))


# ---------------------------------------------------------------- backward
def bwd_core(net, pat, inv_n, c, k, s, ys, e, lam_c):
    """the VJP of step k with the S4D outputs ys (L, B, H) from the tape and their later-step cotangent parts e (L, B, H), given the cotangent
    lam_c of the carry after the step: (cotangent of the carry before the step, cotangent of every S4D output (L, B, H))"""
    L = ys.shape[0]
    (x, aux), pre_vjp = jax.vjp(monitor, net, pat, c, k, s)
    us = []
    for l in range(L):
        us.append(x)
        x = layer(layer_at(net.layers, l), net.mH, net.nH, net.gate[l], x, ys[l])
    _, post_vjp = jax.vjp(post, net, pat, c, s, x, aux)
    g_post = post_vjp((lam_c, inv_n * s.w))
    cx = g_post[4]
    cy = [None] * L
    for l in reversed(range(L)):
        _, lvjp = jax.vjp(layer_tapped, layer_at(net.layers, l), net.mH, net.nH, net.gate[l], us[l], ys[l], e[l])
        g_l = lvjp(cx)
        cx, cy[l] = g_l[4], g_l[5]
    g_pre = pre_vjp((cx, g_post[5]))
    c_bar = jax.tree_util.tree_map(jnp.add, g_post[2], g_pre[2])
    return c_bar, jnp.stack(cy)


def _s4d_e(lam_h, dB):
    """e = Re sum conj(lam_h) dB: (L, B, H)"""
    return jnp.real(jnp.sum(jnp.conj(lam_h) * dB[:, None], -1))


def _bwd_body(carry, xs):
    net, pat, inv_n, lc, lh = carry
    k, s, c, ys = xs
    e = _s4d_e(lh, net.layers.dB)
    c_bar, cy = bwd_core(net, pat, inv_n, c, k, s, ys, e, lc)
    lh2 = (lh + 2.0 * cy[..., None] * jnp.conj(net.layers.C)[:, None]) * jnp.conj(net.layers.dA)[:, None]
    return (net, pat, inv_n, c_bar, lh2), (lc, e)


def _bblk_inner(carry, xs):
    net, pat, bm, inv_n, E, lc, cybuf = carry
    i, k, s, c, ys = xs
    e = (lax.dynamic_index_in_dim(E, i, -1, keepdims=False)
         + jnp.einsum("lbhm,lhm->lbh", cybuf, lax.dynamic_index_in_dim(bm.Kt, i, 3, keepdims=False), precision=HI))
    c_bar, cy = bwd_core(net, pat, inv_n, c, k, s, ys, e, lc)
    return (net, pat, bm, inv_n, E, c_bar, lax.dynamic_update_index_in_dim(cybuf, cy, i, -1)), (lc, e)


def _bblk_body(carry, xs):
    """one block backward: e of its steps from the cotangent of the state at its end (E = Re sum conj(lam) Wu) and its later steps' S4D
    output cotangents; the state cotangent at its start lam conj(dA^T) + conj(2 sum_m cy_m C dA^(m+1))"""
    net, pat, bm, inv_n, lc, lh = carry
    ks, sb, cb, ysb = xs
    T = ks.shape[0]
    E = jnp.real(jnp.einsum("lbhn,lhnt->lbht", jnp.conj(lh), bm.Wu, precision=HI))
    cybuf = jnp.zeros(E.shape, E.dtype)
    (_, _, _, _, _, lc2, cybuf), out = lax.scan(_bblk_inner, (net, pat, bm, inv_n, E, lc, cybuf), (jnp.arange(T), ks, sb, cb, ysb),
                                                reverse=True, unroll=True)
    G = jnp.einsum("lbht,lhnt->lbhn", 2.0 * cybuf.astype(lh.dtype), bm.Wb, precision=HI)
    return (net, pat, bm, inv_n, lc2, lh * jnp.conj(bm.AT)[:, None] + jnp.conj(G)), out


def _blocked(a, T):
    """(N, ...) -> (N / T, T, ...)"""
    return a.reshape((a.shape[0] // T, T) + a.shape[1:])


def _backward(net, pat, steps, tape, T=0, unroll=1):
    """backward sweep of one network from the forward tape: Adjoint. T > 0: blocked S4D adjoint"""
    B, N = pat.mu.shape[0], nsteps(steps)
    L, H, NS = net.layers.dA.shape
    dt = pat.mu.dtype
    lc0 = zeros_like_carry(carry0(B, net.lam.shape[0], dt))
    lh0 = jnp.zeros((L, B, H, NS), complex_of(dt))
    inv_n = jnp.asarray(1.0 / N, dt)
    ks, ys = jnp.arange(N), jnp.moveaxis(tape.ys, 0, 1)                 # ys: (N, L, B, H), the steps first
    if T:
        bm = block_mats(net.layers, T)
        xs = (ks.reshape(N // T, T), blocks(steps, T), Carry(*[_blocked(a, T) for a in tape.c]), _blocked(ys, T))
        _, (lam, e) = lax.scan(_bblk_body, (net, pat, bm, inv_n, lc0, lh0), xs, reverse=True, unroll=unroll)
        lam, e = jax.tree_util.tree_map(_flat, lam), _flat(e)
    else:
        _, (lam, e) = lax.scan(_bwd_body, (net, pat, inv_n, lc0, lh0), (ks, steps, tape.c, ys), reverse=True, unroll=unroll)
    return Adjoint(lam=lam, e=jnp.moveaxis(e, 1, 0))


backward = jax.jit(_backward, static_argnames=("T", "unroll"))

