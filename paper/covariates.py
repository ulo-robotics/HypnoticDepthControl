"""Covariate check: UKF-MM and MPC-MM initialized with the patient's covariates, i.e. given the patient's individual PK instead of the
nominal one. The plant's PK is NOM * exp(iiv + shift): iiv ~ 20% lognormal per parameter (the inter-individual variability, what a covariate
model predicts) and the scenario's correlated PK shift (a model error, unknown to any covariate model). A controller with covariate knowledge
f uses NOM * exp(f * iiv): f = 1 is a perfect covariate model, f = 0.5 one that explains half of the deviation. The shift, C50, the Hill
shape, the delay and the noise stay unknown. Patients: the first n of each validation scenario of the 30-minute evaluation
(the same key and order as evaluate.py, sampled with its batch size 2000), so Lethe and the nominal-PK baselines come from the stored
results.

  python paper/covariates.py [--run runs/finetune] [--n 2000] [--workers 12] [--scenarios a,b]

reads RUN/eval30/ and RUN/eval30_mpc/ (the per-patient results of Lethe and the nominal-PK baselines), writes RUN/paper/covariates/<scenario>.npz
and RUN/paper/covariates.json."""
import argparse, json, os, sys, time
from functools import partial
import numpy as np
import jax

from lethe.core import model as M
from lethe.baselines import classic, mpc, parallel
from lethe.evaluate import EVAL_KEY

NOMV = np.array([M.NOM[k] for k in M.PK_KEYS])


class CovMixin:
    """per-patient PK in UKF-MM's estimator, Smith predictor and LQR (rho: UKF-MM's)"""
    cov_f, cov_rho = 1.0, 2e-4

    def reset(self, pts, plant):
        super().reset(pts, plant)
        B = self.B
        pk = NOMV * np.exp(self.cov_f * pts["iiv"])
        dm = int(self.dset.max())
        Ad, Bd, K, Ap, Gp = [], [], [], [], []
        for p in pk:
            a, b = M.discretize(*M.pk_matrices(dict(zip(M.PK_KEYS, p))))
            Ad.append(a); Bd.append(b[:, 0]); K.append(classic.lqr_gain(a, b, rho=self.cov_rho))
            pw = [np.linalg.matrix_power(a, m) for m in range(dm + 1)]
            Ap.append(pw)
            Gp.append([np.stack([pw[m - 1 - s] @ b[:, 0] for s in range(m)]) if m else np.zeros((0, 4)) for m in range(dm + 1)])
        r = self.row
        self.Ad_r, self.Bd_r, self.K_r = np.asarray(Ad)[r], np.asarray(Bd)[r], np.asarray(K)[r]
        self.cl1_r = pk[r, M.PK_KEYS.index("Cl1")]
        self.Ap_r = [np.asarray([Ap[i][m] for i in range(B)])[r] for m in range(dm + 1)]
        self.Gp_r = [np.asarray([Gp[i][m] for i in range(B)])[r] for m in range(dm + 1)]
        self.Ad_pat, self.Bd_pat = np.asarray(Ad), np.asarray(Bd)

    def estimate(self, k, y):
        """classic.UKFMM.estimate with the row's own matrices"""
        B, Mo = self.B, len(self.dset)
        j = np.maximum(0, k - self.d)
        adv = k - self.d >= 1
        up = self.uh[self.row, np.maximum(j - 1, 0)]
        pts, w = self._sigma(self.z, self.P)
        pp = pts.copy()
        pp[..., :4] = np.einsum("rsi,rji->rsj", pts[..., :4], self.Ad_r) + self.Bd_r[:, None, :] * up[:, None, None]
        zp = w @ pp
        dd = pp - zp[:, None]
        Pp = np.swapaxes(dd * w[:, None], 1, 2) @ dd + self.Q
        zp, Pp = np.where(adv[:, None], zp, self.z), np.where(adv[:, None, None], Pp, self.P)
        pts2, _ = self._sigma(zp, Pp)
        pts = np.where(adv[:, None, None], pts2, pts)
        ys = self._h(pts)
        yp = ys @ w
        dy = ys - yp[:, None]
        R = self.Rm
        Pyy = (dy ** 2) @ w + R
        Kg = ((w * dy)[:, None, :] @ (pts - zp[:, None]))[:, 0] / Pyy[:, None]
        nu = np.repeat(y, Mo) - yp
        self.ll = self.lam * self.ll - 0.5 * (np.log(2 * np.pi * Pyy) + nu ** 2 / Pyy).reshape(B, Mo)
        if self.adapt:
            self.Rm = (1 - self.beta) * R + self.beta * np.maximum(nu ** 2 - (Pyy - R), self.rmin)
        self.z, self.P = zp + Kg * nu[:, None], Pp - Kg[:, :, None] * Kg[:, None, :] * Pyy[:, None, None]
        if k > 0 or self.est_gamma:
            self.z[:, :4] = np.maximum(self.z[:, :4], 0)
        m = np.minimum(k, self.d)
        xf = self.z[:, :4].copy()
        for mv in np.unique(m[m > 0]):
            i = np.flatnonzero(m == mv)
            U = self.uh[self.row[i, None], j[i, None] + np.arange(mv)]
            xf[i] = np.einsum("ri,rji->rj", self.z[i, :4], self.Ap_r[mv][i]) + np.einsum("rs,rsj->rj", U, self.Gp_r[mv][i])
        p = np.exp(self.ll - self.ll.max(1, keepdims=True))
        p = np.maximum(p / p.sum(1, keepdims=True), self.pfloor)
        return xf, self.z[:, 4], M.GAMMA, p / p.sum(1, keepdims=True)


class UKFMMcov(CovMixin, classic.UKFMM):
    def __init__(self, f=1.0, **kw):
        super().__init__(**kw)
        self.cov_f = f

    def act(self, k, y):
        B, Mo = self.B, len(self.dset)
        xf, lc, g, p = self.estimate(k, y)
        cr = M.cref(np.exp(lc), g, M.E0, M.EMAX, np.repeat(self.tgt[:, k], Mo))
        um = (self.cl1_r * cr - np.sum(self.K_r * (xf - cr[:, None]), 1)).reshape(B, Mo)
        u = np.clip(np.sum(p * um, 1), 0, M.UMAX)
        self.uh[:, k] = u
        return u


def mpcmm_nlp_cov(n_obs, blocks=mpc.BLOCKS, rho=0.0, safe_w=2.0, beta=2.0, gamma=M.GAMMA, e0=M.E0, emax=M.EMAX):
    """mpc.mpcmm_nlp with the patient's (Ad, Bd) as parameters: p = [the nominal one's p; Ad (16, column-major); Bd (4)]"""
    ca = mpc._ca()
    nb, Np = len(blocks), sum(blocks)
    u, p = ca.SX.sym("u", nb), ca.SX.sym("p", 6 * n_obs + 2 + 20)
    tgt, up = p[6 * n_obs], p[6 * n_obs + 1]
    A, Bv = ca.reshape(p[6 * n_obs + 2: 6 * n_obs + 18], 4, 4), p[6 * n_obs + 18: 6 * n_obs + 22]
    J = rho * ((u[0] - up) ** 2 + ca.sumsqr(u[1:] - u[:-1]))
    for m in range(n_obs):
        x = p[4 * m: 4 * m + 4]
        c50g = ca.exp(gamma * p[4 * n_obs + m])
        pm = p[5 * n_obs + m]
        Jm = 0
        for j, L in enumerate(blocks):
            for _ in range(L):
                x = ca.mtimes(A, x) + Bv * u[j]
                ceg = ca.fmax(x[3], 1e-12) ** gamma
                b = e0 - emax * ceg / (ceg + c50g)
                Jm += (b - tgt) ** 2 + safe_w * mpc.softplus_low(b, beta) ** 2
        J += pm * Jm / Np
    return dict(x=u, p=p, f=J), nb


class MPCMMcov(CovMixin, mpc.MPCMM):
    def __init__(self, f=1.0, blocks=mpc.BLOCKS, rho_mpc=1e-3, tol=1e-6, **kw):
        super().__init__(blocks=blocks, rho_mpc=rho_mpc, tol=tol, **kw)
        self.cov_f = f
        ca = mpc._ca()
        prob, self.nb = mpcmm_nlp_cov(len(self.dset), blocks, rho_mpc)
        self.S = ca.nlpsol("mpcmm_cov", "ipopt", prob, dict(mpc.OPTS, **{"ipopt.tol": tol, "ipopt.max_iter": 100,
                                                                           "ipopt.mu_strategy": "adaptive"}))
        self._maps = {}

    def act(self, k, y):
        B, Mo = self.B, len(self.dset)
        xf, lc, _, p = self.estimate(k, y)
        xf = np.maximum(xf, 0.0)
        AB = np.concatenate([self.Ad_pat.transpose(0, 2, 1).reshape(B, 16), self.Bd_pat], 1)     # column-major Ad
        P = np.concatenate([xf.reshape(B, Mo * 4), lc.reshape(B, Mo), p, self.tgt[:, k][:, None], self.u_prev[:, None], AB], 1).T
        w0 = np.concatenate([self.w[1:], self.w[-1:]], 0)
        r = self.F(x0=self.w, p=P, lbx=self.lbx, ubx=self.ubx)
        w = np.array(r["x"])
        self.w = np.where(np.isfinite(w), w, w0)
        u = np.clip(self.w[0], 0.0, M.UMAX)
        self.u_prev = u
        self.uh[:, k] = u
        return u


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/finetune")
    ap.add_argument("--n", type=int, default=2000, help="patients per scenario (the first n of the evaluation's 2000)")
    ap.add_argument("--workers", type=int, default=12, help="processes for MPC-MM")
    ap.add_argument("--scenarios", default="", help="comma-separated subset (writes covariates_part.json)")
    args = ap.parse_args()
    n, workers, only = args.n, args.workers, (args.scenarios.split(",") if args.scenarios else None)
    R = os.path.join(args.run, "paper")
    os.makedirs(f"{R}/covariates", exist_ok=True)
    out = {}
    names = list(M.VAL)
    for si, s in enumerate(names):
        sp = M.VAL[s][0]
        if only and s not in only:
            continue
        key = jax.random.fold_in(EVAL_KEY, si)
        pn = M.to_numpy(M.sample_patients(key, 2000, sp))
        pn["iiv"] = np.asarray(M.PK_CV * jax.random.normal(jax.random.split(key, 16)[1], (2000, 7)), np.float64)   # sample_patients' iiv draw
        dev = np.abs(NOMV * np.exp(pn["iiv"]) / pn["pk"] - 1).max()
        print(f"{s}: max relative |NOM exp(iiv) / pk - 1| = {dev:.1e} (pk_shift {sp['pk_shift']})", flush=True)
        assert sp["pk_shift"] > 0 or dev < 1e-5, s
        pn = M.take(pn, np.arange(n))
        ref = np.load(f"{args.run}/eval30/{s}.npz")
        refm = np.load(f"{args.run}/eval30_mpc/{s}.npz")
        res = {"policy": ref["policy__J"][:n], "UKF-MM": ref["UKF-MM__J"][:n], "MPC-MM": refm["MPC-MM__J"][:n],
               "MHE+NMPC": refm["MHE+NMPC__J"][:n]}
        viol = {"policy": ref["policy__viol"][:n], "UKF-MM": ref["UKF-MM__viol"][:n], "MPC-MM": refm["MPC-MM__viol"][:n],
                "MHE+NMPC": refm["MHE+NMPC__viol"][:n]}
        extra = {c: {k: (ref if c in ("policy", "UKF-MM") else refm)[f"{c}__{k}"][:n] for k in ("Jm", "low")}
                 for c in ("policy", "UKF-MM", "MPC-MM", "MHE+NMPC")}
        if si == 0 and not only:                          # f = 0 must reproduce the nominal-PK controllers exactly
            b, u = M.run_numpy(UKFMMcov(f=0.0), M.take(pn, np.arange(20)))
            print("check UKF-MM f=0: max |dJ| %.2e" % np.max(np.abs(M.metrics(b, u, pn["tgt"][:20])["J"] - res["UKF-MM"][:20])), flush=True)
            b, u = parallel.run_parallel(partial(MPCMMcov, f=0.0), M.take(pn, np.arange(8)), 8)
            print("check MPC-MM f=0: max |dJ| %.2e" % np.max(np.abs(M.metrics(b, u, pn["tgt"][:8])["J"] - res["MPC-MM"][:8])), flush=True)
        for name, make, par in (("UKF-MM cov0.5", partial(UKFMMcov, f=0.5), False), ("UKF-MM cov1", partial(UKFMMcov, f=1.0), False),
                                ("MPC-MM cov1", partial(MPCMMcov, f=1.0), True)):
            t = time.time()
            b, u = parallel.run_parallel(make, pn, workers) if par else M.run_numpy(make(), pn)
            m = M.metrics(b, u, pn["tgt"])
            res[name], viol[name], extra[name] = m["J"], m["viol"], {"Jm": m["Jm"], "low": m["low"]}
            print(f"{s:13s} {name:14s} {time.time() - t:6.0f} s", flush=True)
        np.savez(f"{R}/covariates/{s}.npz", **{f"{c}__{k}": v for c in res for k, v in
                                                     (("J", res[c]), ("viol", viol[c]), ("Jm", extra[c]["Jm"]), ("low", extra[c]["low"]))})
        out[s] = {c: dict(J=float(np.mean(v)), viol=float(np.mean(viol[c])),
                          d=float(np.mean(res["policy"] - v)), ci=float(1.96 * np.std(res["policy"] - v, ddof=1) / np.sqrt(n)),
                          win=float(np.mean(res["policy"] < v))) for c, v in res.items()}
        print(s, {c: round(v["J"], 2) for c, v in out[s].items()}, flush=True)
        json.dump(out, open(f"{R}/covariates{'_part' if only else ''}.json", "w"), indent=1)
    if only:
        sys.exit(0)
    st = {c: M.score({s: dict(J=out[s][c]["J"], viol=out[s][c]["viol"]) for s in names}) for c in out[names[0]]}
    print("score:", {c: round(v, 2) for c, v in st.items()})
    json.dump(dict(per=out, score=st, n=n), open(f"{R}/covariates.json", "w"), indent=1)
