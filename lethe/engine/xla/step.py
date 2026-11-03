"""
One closed-loop step for a batch of rows (patients, or one patient's steps in the gradient assembly), in three stages so that the backward
sweep can take its VJP stage by stage:

  monitor    true BIS, delay line, measured BIS, effect-site estimate, the rescaled physics modes, features, encoder
  layer      one residual layer after its S4D output: GELU, gated mixing, residual, layer norm over the active channels (skipped beyond the
             network's depth)
  post       head / decoder, infusion, physics modes, plant, drift, training cost

The S4D recurrence itself (h' = dA h + dB u, ys = 2 Re sum C h') stays with the sweeps: the forward sweep runs it, the backward sweep and the
assembly replay the step with the S4D outputs taken from the tape (s4d_tap). The math is policy.act and the plant of policy.rollout.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from lethe.core import model as M
from lethe.core import policy as P
from lethe.engine.xla.layout import HI, Carry


class Aux(NamedTuple):
    """what post needs from monitor"""
    b: jax.Array         # (B,): true BIS
    buf: jax.Array       # (B, DP): the new delay line
    w: jax.Array         # (B,): effect-site estimate from the measured BIS
    hp: jax.Array        # (B, P) complex: physics modes rescaled to the current reference


def monitor(net, pat, c, k, s):
    """(encoder output (B, H), Aux). k: step index (scalar, or one per row); s: this step's Steps rows"""
    dt = pat.mu.dtype
    t = k.astype(dt) * M.DT
    l50 = (pat.mu + c.ou + pat.stim * (t > pat.stim_t).astype(dt) + pat.stim2 * (t > pat.stim_t2).astype(dt)
           + pat.stim3 * (t > pat.stim_t3).astype(dt))
    b = M.hill(c.x[:, 3], l50, pat.gamma, pat.e0, pat.emax)
    first = jnp.reshape(k == 0, (-1, 1))                                 # the first reading fills the whole delay line
    buf = jnp.where(first, b[:, None], jnp.concatenate([b[:, None], c.buf[:, :-1]], 1))
    y = jnp.sum(buf * pat.dmask, -1) + s.sy * s.z0
    w = P.w_from_y(y)
    feat = jnp.stack([(y - s.tg) / 50, c.up / (s.r * M.UMAX), w / (s.r * M.C50N) - 1.0], -1)
    x = jnp.matmul(feat, net.enc_W, precision=HI) + net.enc_b
    return x, Aux(b=b, buf=buf, w=w, hp=c.hp * s.rho[:, None])


def layer(lp, mH, nH, gate, x, ys):
    """residual layer with S4D output ys (lp: one layer of Layers)"""
    a = jax.nn.gelu(ys + lp.D * x)
    v = x + (jnp.matmul(a, lp.Wv, precision=HI) + lp.bv) * jax.nn.sigmoid(jnp.matmul(a, lp.Wg, precision=HI) + lp.bg)
    m = jnp.sum(v * mH, -1, keepdims=True) / nH
    dv = (v - m) * mH
    out = dv / jnp.sqrt(jnp.sum(dv * dv, -1, keepdims=True) / nH + 1e-5) * lp.g + lp.beta
    return jnp.where(gate > 0, out, x)


def post(net, pat, c, s, x, aux):
    """(Carry after the step, training cost of the step (B,))"""
    o_head = jnp.matmul(jax.nn.gelu(jnp.matmul(x, net.hW, precision=HI) + net.hb), net.hdec, precision=HI)
    o = jnp.where(net.head > 0, o_head, jnp.matmul(x, net.dec_lin, precision=HI)) + net.dec_b
    u_lin = net.u0 + jnp.real(jnp.sum(net.Cout * aux.hp, -1))
    u = jnp.clip(s.r * (u_lin + 10.0 * o), 0.0, M.UMAX)
    hp = net.lam * aux.hp + net.Bu * (u / s.r)[:, None] + net.Bw * (aux.w / s.r)[:, None]
    x2 = jnp.einsum("bij,bj->bi", pat.Ad, c.x, precision=HI) + pat.Bd * u[:, None]
    ou = pat.a_ou * c.ou + pat.osc * s.z1
    cost = ((aux.b - s.tg) / 10) ** 2 + net.w_low * (jax.nn.relu(40 - aux.b) / 10) ** 2 + net.rho_u * (u / M.UMAX) ** 2
    return Carry(x=x2, ou=ou, up=u, hp=hp, buf=aux.buf), cost


def s4d_out(lp, h):
    """the S4D output of state h (B, H, NS): 2 Re sum C h"""
    return 2.0 * jnp.real(jnp.sum(lp.C * h, -1))


@jax.custom_vjp
def s4d_tap(u, ys, e, K0):
    """a replayed S4D layer: value = the taped output ys; the cotangent of its input u is K0 * (cotangent of ys) + e (e: the part from later
    steps, from the backward sweep)"""
    return ys


def _tap_fwd(u, ys, e, K0):
    return ys, (e, K0)


def _tap_bwd(res, ct):
    e, K0 = res
    return ct * K0 + e, ct, jnp.zeros_like(e), jnp.zeros_like(K0)


s4d_tap.defvjp(_tap_fwd, _tap_bwd)


def layer_tapped(lp, mH, nH, gate, u, ys, e):
    """residual layer on a replayed S4D (input u, taped output ys, later-step cotangent part e)"""
    return layer(lp, mH, nH, gate, u, s4d_tap(u, ys, e, lp.K0))


def sums_row(cost, b, s):
    """one step's contributions to Sums"""
    return cost * s.w, (b - s.tg) ** 2, jax.nn.relu(40 - b) ** 2
