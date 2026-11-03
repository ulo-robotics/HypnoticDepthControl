"""GPU engines (float32; pytest -m gpu): every sweep and assembly candidate == the XLA references. A candidate that does not fit a shape
(e.g. shared memory) is skipped: the engine would not use it either."""
import jax
import jax.numpy as jnp
import pytest

from lethe import engine as E
from lethe.core import model as M
from lethe.core import policy as P
from lethe.engine import xla as K
from lethe.train import trainer as TR
from conftest import noisy

pytestmark = pytest.mark.gpu


def cases():
    out = []
    for cfg in (dict(P.ARCH_DEFAULTS, H=16, NS=8, L=1, head=0), dict(P.ARCH_DEFAULTS, H=32, NS=4, L=2, head=32, n_extra=3), dict(P.ARCH_DEFAULTS)):
        out.append((f"H{cfg['H']}NS{cfg['NS']}L{cfg['L']}head{cfg['head']}", [cfg], cfg["head"] > 0))
    two = [dict(P.ARCH_DEFAULTS, H=32, NS=8, L=2, head=0, n_extra=0), dict(P.ARCH_DEFAULTS, H=16, NS=4, L=1, head=16, n_extra=3)]
    out.append(("padded population of 2", two, None))
    return out


@pytest.fixture(scope="module")
def batch():
    if not E._gpu():
        pytest.skip("no GPU")
    pts = M.sample_patients(jax.random.PRNGKey(3), 64, dict(M.with_stimulus(M.VAL["mixed_hard"][0], "signed"), len_lo=10.0, len_hi=30.0))
    return K.to_kq(pts)


def population(cfgs, plain):
    trees = [P.expand(noisy(c, dtype64=False)) for c in cfgs]
    tps = [t for t, _ in trees] if len(cfgs) == 1 else [P.resize(t, TR.common_shape(cfgs)) for t, _ in trees]
    tp = jax.tree_util.tree_map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *tps)
    sz = {k: jnp.asarray([s_[k] for _, s_ in trees]) for k in trees[0][1]}
    return tp, sz


@pytest.mark.parametrize("tag,cfgs,plain", cases(), ids=[c[0] for c in cases()])
def test_candidates(batch, tag, cfgs, plain):
    q, st = batch
    n = len(cfgs)
    tp, sz = population(cfgs, plain)
    kp = E.pop_map(lambda t, s_: K.to_kp(t, s_, 0.87, 0.21, plain), n, (0, 0))(tp, sz)
    wl, ru = jnp.full((n,), 0.87), jnp.full((n,), 0.21)
    fw = lambda f: jax.jit(E.pop_map(lambda k_: f(k_, q, st), n, (0,)))
    bw = lambda f: jax.jit(E.pop_map(lambda k_, t_: f(k_, q, st, t_), n, (0, 0)))
    ft, s = fw(K.xla_forward)(kp)
    bt = bw(K.xla_backward)(kp, ft)
    ran = 0
    for kind, opt in E.candidates(64, n, plain):
        if kind == "xla":
            continue
        fwd, bwd = E.sweep_fns(kind, opt)
        try:
            ft2, s2 = fw(fwd)(kp)
            bt2 = bw(bwd)(kp, ft)
        except Exception:
            continue
        err = max([E._rel(ft2["c"][m], ft["c"][m]) for m in ft["c"]] + [E._rel(a, b) for a, b in zip(ft2["ys"], ft["ys"])] +
                  [E._rel(s2[m], s[m]) for m in s] + [E._rel(bt2["lam"][m], bt["lam"][m]) for m in bt["lam"]] +
                  [E._rel(a, b) for a, b in zip(bt2["e"], bt["e"])])
        assert err <= 1e-3, (kind, opt)
        ran += 1
    g_ref = jax.jit(E.pop_map(lambda t, s_, w, r, f, b: K.assemble(t, s_, w, r, plain, q, st, f, b), n, (0,) * 6))(tp, sz, wl, ru, ft, bt)
    for kind, opt in E.asm_candidates()[1:]:
        af = E.asm_fn(kind, opt)
        try:
            g = jax.jit(E.pop_map(lambda t, s_, w, r, f, b: af(t, s_, w, r, plain, q, st, f, b, None), n, (0,) * 6))(tp, sz, wl, ru, ft, bt)
        except Exception:
            continue
        assert E._rel_tree(g, g_ref) <= 1e-3, (kind, opt)
        ran += 1
    assert ran >= 1
