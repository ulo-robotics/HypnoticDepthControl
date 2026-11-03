"""The adjoint engine (XLA reference): per-patient gradients equal raw autodiff, blocked sweeps equal per-step ones, padding is exact"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lethe import engine as E
from lethe.core import policy as P
from lethe.engine import xla as K
from lethe.train import trainer as TR
from conftest import noisy, rel

SHAPES = [dict(H=8, NS=4, L=2, head=8, n_extra=3), dict(H=16, NS=8, L=1, head=0)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"H{s['H']}L{s['L']}head{s['head']}")
@pytest.mark.parametrize("layout", ["plain", "masked"])
def test_adjoint_equals_raw(cfg, pts, shape, layout):
    c = dict(cfg, **shape)
    tp, sz = P.expand(noisy(c))
    g_ref = E.raw_grads(tp, sz, pts, 0.87, 0.21)[0]
    g_adj = K.adjoint_grads(tp, sz, pts, 0.87, 0.21, plain=(c["head"] > 0) if layout == "plain" else None)[0]
    assert rel(g_adj, g_ref) <= 1e-8                                    # float64 rounding through 360 steps


@pytest.mark.parametrize("T", [10, 20])
def test_blocked_sweeps(cfg, pts, T):
    tp, sz = P.expand(noisy(dict(cfg, **SHAPES[0])))
    kp, (q, st) = K.to_kp(tp, sz, 0.87, 0.21, None), K.to_kq(pts)
    ft, s0 = K.xla_forward(kp, q, st)
    bt = K.xla_backward(kp, q, st, ft)
    ft2, s2 = K.xla_forward(kp, q, st, T=T)
    bt2 = K.xla_backward(kp, q, st, ft, T=T)
    assert max(rel(ft2, ft), rel(s2, s0), rel(bt2, bt)) <= 1e-10


def test_adjoint_with_blocked_sweeps(cfg, pts):
    tp, sz = P.expand(noisy(dict(cfg, **SHAPES[0])))
    g = K.adjoint_grads(tp, sz, pts, 0.87, 0.21, forward=lambda *a, **k_: K.xla_forward(*a, **k_, T=20),
                        backward=lambda *a: K.xla_backward(*a, T=20))[0]
    assert rel(g, E.raw_grads(tp, sz, pts, 0.87, 0.21)[0]) <= 1e-8


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"H{s['H']}L{s['L']}head{s['head']}")
def test_padding_is_exact(cfg, pts, shape):
    c = dict(cfg, **shape)
    cp = noisy(c)
    tp, sz = P.expand(cp)
    big = dict(H=sz["H"] + 8, NS=sz["NS"] + 4, L=sz["L"] + 1, HD=16, P=5 + c.get("n_extra", 0) + 3)
    tpb = jax.tree_util.tree_map(jnp.asarray, P.resize(tp, big))
    own = TR.own_shape(c)
    _, b0, _ = P.rollout(cp, pts)
    _, b1, _ = P.rollout(tpb, pts, sz=sz)
    assert float(jnp.max(jnp.abs(b1 - b0))) <= 1e-10
    g0 = E.raw_grads(tp, sz, pts, 0.87, 0.21)[0]
    for gf in (E.raw_grads, lambda *a: K.adjoint_grads(*a, plain=None)):
        g1 = gf(tpb, sz, pts, 0.87, 0.21)[0]
        per = [jax.tree_util.tree_map(lambda x: np.asarray(x[i]), g1) for i in range(pts["mu"].shape[0])]
        g1c = jax.tree_util.tree_map(lambda *x: np.stack(x), *[P.resize(g, own) for g in per])          # the active entries
        pad = max(float(np.max(np.abs(x - y))) for g in per                                             # what cropping + re-padding removes
                  for x, y in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(P.resize(P.resize(g, own), big))))
        assert rel(g1c, g0) <= 1e-8
        assert pad == 0.0
