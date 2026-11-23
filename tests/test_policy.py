"""The policy and its closed loop: prior compensator, rollout vs the numpy plant, gradients"""
import jax
import jax.numpy as jnp
import numpy as np

from lethe.core import model as M
from lethe.core import policy as P
from lethe.train import train as T
from lethe.train import trainer as TR
from conftest import f64


class PriorNumpy:
    """the prior design run as a plain linear compensator (complex modal state), target-normalized like the policy"""

    def __init__(self, cfg):
        self.d = P.design_prior(*(cfg[k] for k in ("q_theta", "sy_scale", "rho_lqr", "q_x")))

    def reset(self, pts, plant):
        self.tgt = pts["tgt"]
        self.h = np.zeros((pts["mu"].shape[0], len(self.d["lam"])), complex)

    def act(self, k, y):
        wr = lambda t: M.cref(M.C50N, M.GAMMA, M.E0, M.EMAX, t)
        tg, tgp = self.tgt[:, k], self.tgt[:, max(k - 1, 0)]
        r = wr(tg) / M.C50N
        self.h *= (wr(tgp) / wr(tg))[:, None]
        s = np.clip((M.E0 - y) / M.EMAX, 1e-3, 1 - 1e-3)
        w = np.minimum(M.C50N * np.sqrt(s / (1 - s)), 3 * M.C50N)
        u = np.clip(r * (self.d["u0"] + np.real(self.h @ self.d["Cout"])), 0, M.UMAX)
        self.h = self.h * self.d["lam"] + np.outer(u / r, self.d["Bu"]) + np.outer(w / r, self.d["Bw"])
        return u


def test_untrained_policy_is_the_prior(cfg, pts, pn, p0):
    _, _, u = P.rollout(p0, pts)
    _, un = M.run_numpy(PriorNumpy(cfg), pn)
    assert np.max(np.abs(np.asarray(u).T - un)) <= 1e-3                 # mg/min; the modes are stored in float32


def test_rollout_equals_numpy_plant(cfg, pts, pn, p1):
    _, b, _ = P.rollout(p1, pts)
    bn, un = M.run_numpy(P.PolicyController(p1), pn)
    assert np.max(np.abs(np.asarray(b).T - bn)) <= 1e-8
    assert np.mean(np.abs(un - M.run_numpy(PriorNumpy(cfg), pn)[1])) >= 0.1   # the residual branch is active


def test_finite_difference(p1):
    ptn = f64(M.sample_patients(jax.random.PRNGKey(4), 8, M.VAL["nominal"][0]))
    f = lambda p: jnp.mean(P.rollout(p, ptn, 0.87, 0.21)[0])
    g = jax.grad(f)(p1)
    leaves, tdef = jax.tree_util.tree_flatten(p1)
    k = jax.random.PRNGKey(1)
    dirs = jax.tree_util.tree_unflatten(tdef, [jax.random.normal(jax.random.fold_in(k, i), x.shape, x.dtype) for i, x in enumerate(leaves)])
    dd = sum(jnp.sum(a * b) for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(dirs)))
    eps = 1e-6
    shift = lambda s: jax.tree_util.tree_map(lambda x, d: x + s * d, p1, dirs)
    fd = (f(shift(eps)) - f(shift(-eps))) / (2 * eps)
    assert abs(dd - fd) / abs(fd) <= 1e-6


def test_per_patient_mean_is_batch_gradient(p1):
    ptn = f64(M.sample_patients(jax.random.PRNGKey(4), 8, M.VAL["nominal"][0]))
    c = dict(T.CONFIG, patient_clip=0.0)
    one = lambda pt: {kk: (v[:, :, None] if kk == "z" else v[None]) for kk, v in pt.items()}
    gp = jax.vmap(jax.grad(lambda p, pt: jnp.mean(P.rollout(p, one(pt), c["w_low"], c["rho_u"])[0])), in_axes=(None, P.patient_axes(ptn)))(p1, ptn)
    ga, _ = TR.aggregate(gp, 0.0)
    gb = jax.grad(lambda p: jnp.mean(P.rollout(p, ptn, c["w_low"], c["rho_u"])[0]))(p1)
    num = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb)))
    den = max(float(jnp.max(jnp.abs(b))) for b in jax.tree_util.tree_leaves(gb))
    assert num / den <= 1e-10


def test_episode_length_is_data(p1):
    """a spec's T sets the steps; a patient's len masks its cost to its own steps (mean over them)"""
    sp = dict(M.VAL["nominal"][0], T=10.0)
    pa = M.sample_patients(jax.random.PRNGKey(5), 4, sp)
    assert pa["z"].shape[0] == M.steps(sp) == 120
    sl = dict(sp, len_lo=2.0, len_hi=10.0)
    pl = f64(M.sample_patients(jax.random.PRNGKey(5), 4, sl))
    loss, b, u = P.rollout(p1, pl)
    full = dict(pl)
    full.pop("len")
    _, b2, _ = P.rollout(p1, full)
    np.testing.assert_array_equal(np.asarray(b), np.asarray(b2))            # the length only weights the cost
    n = np.asarray(pl["len"])
    assert (n >= 24).all() and (n <= 120).all()


def test_load_model_drops_the_time_row(tmp_path, cfg, pts):
    """a model saved with the earlier 4-row encoder (time row held at zero) loads as the same policy; one that used it is refused"""
    import pickle
    p = jax.tree_util.tree_map(np.asarray, P.init_params(jax.random.PRNGKey(2), cfg))
    W = p["extra"]["enc_W"]
    old = jax.tree_util.tree_map(lambda x: x, p)
    old["extra"] = dict(p["extra"], enc_W=np.insert(W, 2, 0.7, axis=0))
    f = tmp_path / "old.pkl"
    pickle.dump(dict(params=old, cfg=cfg), open(f, "wb"))
    np.testing.assert_array_equal(P.load_model(str(f))["params"]["extra"]["enc_W"], W)
    pickle.dump(dict(params=old, cfg=cfg, semantics=dict(time_feature=True, tau_max=0.0)), open(f, "wb"))
    try:
        P.load_model(str(f))
        raise AssertionError("a model that used the time input was loaded")
    except SystemExit:
        pass
