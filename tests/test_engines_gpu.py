"""GPU engines (float32; pytest -m gpu): every sweep and assembly candidate == the XLA references, on the engine's interface (the CUDA
kernels' tape format). A candidate that does not fit a shape (e.g. shared memory) is skipped: the engine would not use it either."""
import jax
import jax.numpy as jnp
import pytest

from lethe import engine as E
from lethe.core import model as M
from lethe.core import policy as P
from lethe.engine import xla as X
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
def pts():
    if not E._gpu():
        pytest.skip("no GPU")
    return M.sample_patients(jax.random.PRNGKey(3), 64, dict(M.with_stimulus(M.VAL["mixed_hard"][0], "signed"), len_lo=10.0, len_hi=30.0))


def population(cfgs):
    trees = [P.expand(noisy(c, dtype64=False)) for c in cfgs]
    tps = [t for t, _ in trees] if len(cfgs) == 1 else [P.resize(t, TR.common_shape(cfgs)) for t, _ in trees]
    tp = jax.tree_util.tree_map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *tps)
    sz = {k: jnp.asarray([s_[k] for _, s_ in trees]) for k in trees[0][1]}
    n = len(cfgs)
    return (tp, sz, jnp.full((n,), 0.87), jnp.full((n,), 0.21)), n


def err_fwd(ft2, s2, ft, s):
    return max([E._rel(ft2["c"][m], ft["c"][m]) for m in ft["c"]] + [E._rel(a, b) for a, b in zip(ft2["ys"], ft["ys"])] +
               [E._rel(s2[m], s[m]) for m in s])


def err_bwd(bt2, bt):
    return max([E._rel(bt2["lam"][m], bt["lam"][m]) for m in bt["lam"]] + [E._rel(a, b) for a, b in zip(bt2["e"], bt["e"])])


@pytest.mark.parametrize("tag,cfgs,plain", cases(), ids=[c[0] for c in cases()])
def test_candidates(pts, tag, cfgs, plain):
    net, n = population(cfgs)
    ft, s = jax.jit(E.fwd_pop(X.forward_k, plain, pts, n))(*net)
    bt = jax.jit(E.bwd_pop(X.backward_k, plain, pts, n))(*net, ft)
    ran = 0
    for kind, opt in E.candidates(64, n, plain):
        if kind == "xla":
            continue
        fwd, bwd = E.sweep_fns(kind, opt)
        try:
            ft2, s2 = jax.jit(E.fwd_pop(fwd, plain, pts, n))(*net)
            bt2 = jax.jit(E.bwd_pop(bwd, plain, pts, n))(*net, ft)
        except Exception:
            continue
        assert err_fwd(ft2, s2, ft, s) <= 1e-3 and err_bwd(bt2, bt) <= 1e-3, (kind, opt)
        ran += 1
    g_ref = jax.jit(E.asm_pop(X.assemble_k, plain, pts, None, n))(*net, ft, bt)
    for kind, opt in E.asm_candidates()[1:]:
        try:
            g = jax.jit(E.asm_pop(E.asm_fn(kind, opt), plain, pts, None, n))(*net, ft, bt)
        except Exception:
            continue
        assert E._rel_tree(g, g_ref) <= 1e-3, (kind, opt)
        ran += 1
    assert ran >= 1
