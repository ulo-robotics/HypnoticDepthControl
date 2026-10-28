#!/usr/bin/env python
"""
Correctness checks.
Float64 (a minute on a CPU):
  1. the untrained policy is exactly the prior compensator (an independent numpy implementation of the observer + LQR in modal form)
  2. the JAX closed loop used for training (policy.rollout) == the numpy reference plant driven step by step, on hard patients
     (delays, target steps, noise steps, PD / PK shifts) with a non-trivial residual branch
  3. the pathwise gradient matches a finite difference
  4. per-patient aggregation without clipping == the gradient of the batch-mean loss
  5. the adjoint gradient (kstep.py, XLA sweeps) == raw autodiff, per patient, for several shapes (head / no head, extra physics modes, H < 16),
     in the plain and in the masked kernel layout
  6. padding is exact: a network zero-padded into a larger shape (more channels, states, layers, head units, physics modes) computes the same
     closed loop, its per-patient gradients equal the unpadded ones on the active entries and are exactly zero on the padding (raw and adjoint)
  7. a population of different configurations (sizes, optimizers, learning-rate rules, loss weights) trained together == each trained alone
  8. the UKF-MM baseline with one observer of delay 0 and fixed noise == UKF+LQR, and the full bank runs (finite, in range)
Float32, GPU only (a fresh process, `python selftest.py --engines`; run automatically when a GPU is present):
  9. every Pallas and CUDA sweep candidate of engine.py == the XLA reference sweeps on every tape, for a single network (three shapes including
     the selected one) and a padded population. A candidate that does not fit a shape (e.g. shared memory) is reported as skipped: the engine
     would not use it either.
"""
import os, subprocess, sys
import numpy as np
import jax
import jax.numpy as jnp

import model as M
import policy as P
import train as T
import kstep as K

ENGINE_MODE = "--engines" in sys.argv
if not ENGINE_MODE:
    jax.config.update("jax_enable_x64", True)
CFG = dict(P.ARCH_DEFAULTS, H=8, NS=4, L=2, head=8)


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


def check(name, err, tol, at_least=False):
    ok = err >= tol if at_least else err <= tol
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {err:.2e} ({'min' if at_least else 'tol'} {tol:.0e})", flush=True)
    return ok


def f64(tree):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if x.dtype == jnp.float32 else x, tree)


def main():
    pts = f64(M.sample_patients(jax.random.PRNGKey(3), 32, M.VAL["mixed_hard"][0]))
    pn = M.to_numpy(pts)
    ok = []

    # 1. untrained policy == prior compensator
    p0 = f64(P.init_params(jax.random.PRNGKey(0), CFG))
    _, b, u = P.rollout(p0, pts)
    bn, un = M.run_numpy(PriorNumpy(CFG), pn)
    ok.append(check("untrained policy == prior compensator (max |du|, mg/min; the modes are stored in float32)", float(np.max(np.abs(np.asarray(u).T - un))), 1e-3))

    # 2. JAX rollout == numpy plant + policy steps, with the residual branch switched on
    k = jax.random.PRNGKey(1)
    p1 = jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(k, x.shape, x.dtype), p0)
    p1["phys"] = p0["phys"]
    _, b, u = P.rollout(p1, pts)
    bn, un = M.run_numpy(P.PolicyController(p1), pn)
    ok.append(check("rollout == numpy plant + policy (max |dBIS|)", float(np.max(np.abs(np.asarray(b).T - bn))), 1e-8))
    ok.append(check("residual branch is active (mean |u - prior u|, mg/min)", float(np.mean(np.abs(un - M.run_numpy(PriorNumpy(CFG), pn)[1]))), 0.1, at_least=True))

    # 3. finite difference along a random direction (smooth patients: nominal scenario)
    ptn = f64(M.sample_patients(jax.random.PRNGKey(4), 8, M.VAL["nominal"][0]))
    f = lambda p: jnp.mean(P.rollout(p, ptn, 0.87, 0.21)[0])
    g = jax.grad(f)(p1)
    leaves, tdef = jax.tree_util.tree_flatten(p1)
    dirs = jax.tree_util.tree_unflatten(tdef, [jax.random.normal(jax.random.fold_in(k, i), x.shape, x.dtype) for i, x in enumerate(leaves)])
    dd = sum(jnp.sum(a * b) for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(dirs)))
    eps = 1e-6
    shift = lambda s: jax.tree_util.tree_map(lambda x, d: x + s * d, p1, dirs)
    fd = (f(shift(eps)) - f(shift(-eps))) / (2 * eps)
    ok.append(check("directional derivative vs finite difference (relative)", float(abs(dd - fd) / abs(fd)), 1e-6))

    # 4. per-patient gradients averaged without clipping == batch gradient
    c = dict(T.CONFIG, patient_clip=0.0)
    gp = jax.vmap(jax.grad(lambda p, pt: jnp.mean(P.rollout(p, {kk: (v[:, :, None] if kk == "z" else v[None]) for kk, v in pt.items()},
                                                            c["w_low"], c["rho_u"])[0])), in_axes=(None, P.PATIENT_AXES))(p1, ptn)
    ga, _ = T.aggregate(gp, 0.0)
    gb = jax.grad(lambda p: jnp.mean(P.rollout(p, ptn, c["w_low"], c["rho_u"])[0]))(p1)
    num = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb)))
    den = max(float(jnp.max(jnp.abs(b))) for b in jax.tree_util.tree_leaves(gb))
    ok.append(check("mean of per-patient gradients == batch gradient (relative)", num / den, 1e-10))

    # 5. adjoint gradient (XLA sweeps) == raw autodiff, per patient
    import engine as E
    import trainer as TR
    rel = lambda ga, gb: max(float(jnp.max(jnp.abs(x - y)) / (jnp.max(jnp.abs(y)) + 1e-30)) for x, y in zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb)))
    noisy = lambda cfg, key=k: jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(key, x.shape, x.dtype), f64(P.init_params(jax.random.PRNGKey(0), cfg)))
    for cfg in (dict(CFG, head=8, n_extra=3), dict(CFG, H=16, NS=8, L=1, head=0)):
        tp, sz = P.expand(noisy(cfg))
        g_ref = E.raw_grads(tp, sz, pts, 0.87, 0.21)[0]
        for plain in (cfg["head"] > 0, None):
            g_adj = K.adjoint_grads(tp, sz, pts, 0.87, 0.21, plain=plain)[0]
            ok.append(check(f"adjoint == raw autodiff, per patient (H{cfg['H']} L{cfg['L']} head{cfg['head']} +{cfg.get('n_extra', 0)} modes, "
                            f"{'masked' if plain is None else 'plain'} layout, relative; float64 rounding through 360 steps)", rel(g_adj, g_ref), 1e-8))

    # 5b. blocked S4D sweeps == per-step sweeps
    tp, sz = P.expand(noisy(dict(CFG, head=8, n_extra=3)))
    kp, (q, st) = K.to_kp(tp, sz, 0.87, 0.21, None), K.to_kq(pts)
    ft, s0 = K.xla_forward(kp, q, st)
    bt = K.xla_backward(kp, q, st, ft)
    for tb in (10, 20):
        ft2, s2 = K.xla_forward(kp, q, st, T=tb)
        bt2 = K.xla_backward(kp, q, st, ft, T=tb)
        err = max(rel(ft2, ft), rel(s2, s0), rel(bt2, bt))
        ok.append(check(f"blocked S4D sweeps (T = {tb}) == per-step sweeps (all tapes, relative)", err, 1e-10))
    g_blk = K.adjoint_grads(tp, sz, pts, 0.87, 0.21, forward=lambda *a, **k_: K.xla_forward(*a, **k_, T=20),
                            backward=lambda *a: K.xla_backward(*a, T=20))[0]
    ok.append(check("adjoint with blocked sweeps == raw autodiff, per patient (relative)", rel(g_blk, E.raw_grads(tp, sz, pts, 0.87, 0.21)[0]), 1e-8))

    # 6. padding is exact
    for cfg in (dict(CFG, head=8, n_extra=3), dict(CFG, H=16, NS=8, L=1, head=0)):
        cp = noisy(cfg)
        tp, sz = P.expand(cp)
        big = dict(H=sz["H"] + 8, NS=sz["NS"] + 4, L=sz["L"] + 1, HD=16, P=5 + cfg.get("n_extra", 0) + 3)
        tpb = jax.tree_util.tree_map(jnp.asarray, P.resize(tp, big))
        own = TR.own_shape(cfg)
        tag = f"H{cfg['H']} L{cfg['L']} head{cfg['head']} -> padded"
        _, b0, _ = P.rollout(cp, pts)
        _, b1, _ = P.rollout(tpb, pts, sz=sz)
        ok.append(check(f"padded network == compact network ({tag}, max |dBIS|)", float(jnp.max(jnp.abs(b1 - b0))), 1e-10))
        g0 = E.raw_grads(tp, sz, pts, 0.87, 0.21)[0]
        for name, gf in (("raw", E.raw_grads), ("adjoint", lambda *a: K.adjoint_grads(*a, plain=None))):
            g1 = gf(tpb, sz, pts, 0.87, 0.21)[0]
            per = [jax.tree_util.tree_map(lambda x: np.asarray(x[i]), g1) for i in range(pts["mu"].shape[0])]
            g1c = jax.tree_util.tree_map(lambda *x: np.stack(x), *[P.resize(g, own) for g in per])        # active entries
            pad_mass = max(float(np.max(np.abs(x - y))) for g in per                                   # what cropping + re-padding removes
                           for x, y in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(P.resize(P.resize(g, own), big))))
            ok.append(check(f"padded per-patient gradients == unpadded ({name}, {tag}, relative)", rel(g1c, g0), 1e-8))
            ok.append(check(f"gradients on the padding are exactly zero ({name}, {tag}, max |g|)", pad_mass, 0.0))

    # 7. a population trained together == each member trained alone
    cfgs = [dict(T.CONFIG, **o, batch=16, patient_clip=4.0) for o in (
        dict(H=8, NS=4, L=2, head=8, n_extra=3, opt="adabelief", sched="exp"),
        dict(H=16, NS=8, L=1, head=0, n_extra=0, opt="lion", sched="cosine", warm=25, w_low=3.0),
        dict(H=8, NS=8, L=1, head=16, n_extra=0, opt="sgd", sched="wsd", ema=0.9))]
    st = [jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64) if np.asarray(x).dtype == np.float32 else x,
                                 TR.new_state(c, jax.random.PRNGKey(i))) for i, c in enumerate(cfgs)]
    key = jax.random.PRNGKey(10_000)
    popn = TR.Population(cfgs, st, key, opt_names=TR.OPT_NAMES, log=lambda m: None, block=3)
    popn.train_to(3)
    together = popn.member_states()
    for i, c in enumerate(cfgs):
        alone = TR.Population([c], [st[i]], key, log=lambda m: None, block=3)
        alone.train_to(3)
        ok.append(check(f"population member == trained alone ({c['opt']}, H{c['H']} NS{c['NS']} L{c['L']} head{c['head']}, 3 iterations, relative)",
                        rel(together[i]["params"], alone.member_states()[0]["params"]), 1e-9))

    # 8. UKF-MM reduces to UKF+LQR
    import baselines as BL
    _, u0 = M.run_numpy(BL.UKFLQR(q_theta=5e-4, rho=2e-4), pn)
    _, u1 = M.run_numpy(BL.UKFMM(dset=(0,), adapt=False), pn)
    ok.append(check("UKF-MM, one observer of delay 0, fixed noise == UKF+LQR (max |du|, mg/min)", float(np.max(np.abs(u0 - u1))), 1e-9))
    _, u2 = M.run_numpy(BL.UKFMM(), pn)
    ok.append(check("UKF-MM bank: inputs finite and in [0, UMAX] (violations)", float(np.sum(~np.isfinite(u2)) + np.sum((u2 < 0) | (u2 > M.UMAX))), 0.0))

    if E._gpu():
        print("engine checks (float32, GPU) in a fresh process:", flush=True)
        ok.append(subprocess.run([sys.executable, os.path.abspath(__file__), "--engines"]).returncode == 0)
    else:
        print("no GPU: Pallas / CUDA engine checks skipped")
    print("ALL OK" if all(ok) else f"{len(ok) - sum(ok)} CHECK(S) FAILED")
    raise SystemExit(0 if all(ok) else 1)


def engines():
    """9. every GPU sweep candidate vs the XLA reference sweeps (float32)"""
    import engine as E
    import trainer as TR
    ok, n_run = [], 0
    noisy = lambda cfg: jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(jax.random.PRNGKey(1), x.shape, x.dtype),
                                               P.init_params(jax.random.PRNGKey(0), cfg))
    pts = M.sample_patients(jax.random.PRNGKey(3), 64, M.VAL["mixed_hard"][0])
    q, st = K.to_kq(pts)
    cases = []
    for cfg in (dict(P.ARCH_DEFAULTS, H=16, NS=8, L=1, head=0), dict(P.ARCH_DEFAULTS, H=32, NS=4, L=2, head=32, n_extra=3), dict(P.ARCH_DEFAULTS)):
        tp, sz = P.expand(noisy(cfg))
        cases.append((f"H{cfg['H']} NS{cfg['NS']} L{cfg['L']} head{cfg['head']}", [tp], [sz], cfg["head"] > 0))
    two = [dict(P.ARCH_DEFAULTS, H=32, NS=8, L=2, head=0, n_extra=0), dict(P.ARCH_DEFAULTS, H=16, NS=4, L=1, head=16, n_extra=3)]
    shape = TR.common_shape(two)
    trees = [P.expand(noisy(c)) for c in two]
    cases.append(("padded population of 2", [P.resize(t, shape) for t, _ in trees], [s_ for _, s_ in trees], None))
    for tag, tps, szs, plain in cases:
        n = len(tps)
        tp = jax.tree_util.tree_map(lambda *a: jnp.stack([jnp.asarray(x) for x in a]), *tps)
        sz = {k: jnp.asarray([s_[k] for s_ in szs]) for k in szs[0]}
        kp = E.pop_map(lambda t, s_: K.to_kp(t, s_, 0.87, 0.21, plain), n, (0, 0))(tp, sz)
        fw = lambda f: jax.jit(E.pop_map(lambda k_: f(k_, q, st), n, (0,)))
        bw = lambda f: jax.jit(E.pop_map(lambda k_, t_: f(k_, q, st, t_), n, (0, 0)))
        ft, s = fw(K.xla_forward)(kp)
        bt = bw(K.xla_backward)(kp, ft)
        for kind, opt in E.candidates(64, n, plain):
            if kind == "xla":
                continue
            fwd, bwd = E.sweep_fns(kind, opt)
            try:
                ft2, s2 = fw(fwd)(kp)
                bt2 = bw(bwd)(kp, ft)
            except Exception as e:
                print(f"[skip] {kind} {opt} {tag}: {type(e).__name__}: {str(e).splitlines()[0][:90]}", flush=True)
                continue
            err = max([E._rel(ft2["c"][m], ft["c"][m]) for m in ft["c"]] + [E._rel(a, b) for a, b in zip(ft2["ys"], ft["ys"])] +
                      [E._rel(s2[m], s[m]) for m in s] + [E._rel(bt2["lam"][m], bt["lam"][m]) for m in bt["lam"]] +
                      [E._rel(a, b) for a, b in zip(bt2["e"], bt["e"])])
            ok.append(check(f"{kind} {opt} sweeps == XLA reference ({tag}, all tapes, relative)", err, 1e-3))
            n_run += 1
    ok.append(check("number of GPU sweep candidates checked", n_run, 1, at_least=True))
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    engines() if ENGINE_MODE else main()
