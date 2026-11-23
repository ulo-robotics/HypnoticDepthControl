"""The XLA adjoint engine: per-patient gradients equal raw autodiff, blocked sweeps equal per-step ones, padding is exact, and the engine's
interface (the CUDA kernels' tape format) gives the same gradients"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lethe import engine as E
from lethe.core import policy as P
from lethe.engine import xla as X
from lethe.train import trainer as TR
from conftest import noisy, rel

SHAPES = [dict(H=8, NS=4, L=2, head=8, n_extra=3), dict(H=16, NS=8, L=1, head=0)]
ids = lambda s: f"H{s['H']}L{s['L']}head{s['head']}"


@pytest.mark.parametrize("shape", SHAPES, ids=ids)
@pytest.mark.parametrize("T", [0, 10])
def test_adjoint_equals_raw(cfg, pts, shape, T):
    tp, sz = P.expand(noisy(dict(cfg, **shape)))
    g_ref, l_ref, J_ref = E.raw_grads(tp, sz, pts, 0.87, 0.21)
    g, l, J = X.adjoint_grads(tp, sz, pts, 0.87, 0.21, T=T)
    assert rel(g, g_ref) <= 1e-8                                        # float64 rounding through 360 steps
    assert rel(l, l_ref) <= 1e-12 and rel(J, J_ref) <= 1e-12


@pytest.mark.parametrize("T", [10, 20])
def test_blocked_sweeps(cfg, pts, T):
    tp, sz = P.expand(noisy(dict(cfg, **SHAPES[0])))
    net, pat, steps = X.net_of(tp, sz, 0.87, 0.21), X.patients_of(pts), X.steps_of(pts)
    ft, s0 = X.forward(net, pat, steps, True, 0)
    bt = X.backward(net, pat, steps, ft, 0)
    ft2, s2 = X.forward(net, pat, steps, True, T)
    bt2 = X.backward(net, pat, steps, ft, T)
    assert max(rel(ft2, ft), rel(s2, s0), rel(bt2, bt)) <= 1e-10


def test_engine_interface(cfg, pts):
    """the sweeps and the assembly through the kernels' tape format == the XLA engine's own gradient"""
    tp, sz = P.expand(noisy(dict(cfg, **SHAPES[0])))
    g, l, J = E.adjoint_grads(tp, sz, pts, 0.87, 0.21)
    g2, l2, J2 = X.adjoint_grads(tp, sz, pts, 0.87, 0.21)
    assert max(rel(g, g2), rel(l, l2), rel(J, J2)) <= 1e-12


@pytest.mark.parametrize("shape", SHAPES, ids=ids)
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
    for gf in (E.raw_grads, X.adjoint_grads):
        g1 = gf(tpb, sz, pts, 0.87, 0.21)[0]
        per = [jax.tree_util.tree_map(lambda x: np.asarray(x[i]), g1) for i in range(pts["mu"].shape[0])]
        g1c = jax.tree_util.tree_map(lambda *x: np.stack(x), *[P.resize(g, own) for g in per])          # the active entries
        pad = max(float(np.max(np.abs(x - y))) for g in per                                             # what cropping + re-padding removes
                  for x, y in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(P.resize(P.resize(g, own), big))))
        assert rel(g1c, g0) <= 1e-8
        assert pad == 0.0
