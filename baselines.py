"""
Baseline controllers on the numpy plant (model.run_numpy), vectorized over a patient batch.

  PID          kp 0.35, Ti 6 min, conditional-integration anti-windup
  OracleCE     knows the true PK, the true C50(t) and the full state: certainty-equivalent LQR (a benchmark, not a controller)
  UKFLQR       UKF on [c1, c2, c3, ce, log C50] (C50 a random walk) + LQR, nominal PK and Hill shape, R = SY^2
  UKFLQR(gamma=True)  same, also estimating log gamma (the fair baseline when the Hill slope is unknown)
  UKFMM        delay-aware multiple-model UKF+LQR (the strongest estimator-based reference): a bank of observers for the monitor delays DSET,
               each estimating the DELAYED state, its control predicted forward over the inputs already applied (a Smith predictor around
               the same LQR), mixed by innovation likelihoods; measurement noise estimated online (Sage-Husa)
"""
import numpy as np
from scipy.linalg import solve_discrete_are

import model as M


def lqr_gain(Ad, Bd, q=1.0, rho=1e-4):
    Q = np.zeros((4, 4)); Q[3, 3] = q
    R = np.array([[rho]])
    P = solve_discrete_are(Ad, Bd, Q, R)
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)[0]


class PID:
    def __init__(self, kp=0.35, ti=6.0):
        self.kp, self.ti = kp, ti

    def reset(self, pts, plant):
        self.tgt = pts["tgt"]
        self.I = np.zeros(pts["mu"].shape[0])

    def act(self, k, y):
        e = y - self.tgt[:, k]
        u_unsat = self.kp * (e + self.I / self.ti)
        u = np.clip(u_unsat, 0, M.UMAX)
        self.I = np.where(u == u_unsat, self.I + e * M.DT, self.I)
        return u


class OracleCE:
    def reset(self, pts, plant):
        self.pts, self.plant = pts, plant
        self.K = np.stack([lqr_gain(Ad, Bd[:, None]) for Ad, Bd in zip(pts["Ad"], pts["Bd"])])
        self.cl1 = pts["pk"][:, M.PK_KEYS.index("Cl1")]

    def act(self, k, y):
        p = self.pts
        cr = M.cref(np.exp(self.plant.logc50(k)), p["gamma"], p["e0"], p["emax"], p["tgt"][:, k])
        u = self.cl1 * cr - np.sum(self.K * (self.plant.x - cr[:, None]), 1)
        return np.clip(u, 0, M.UMAX)


class UKFLQR:
    def __init__(self, gamma=False, q_theta=2e-3, q_x=1e-4, sy=M.SY, p0_theta=0.3, q_gamma=1e-5, p0_gamma=1.0, rho=1e-4):
        self.est_gamma = gamma
        self.Ad, Bd = M.discretize(*M.pk_matrices(M.NOM))
        self.Bd = Bd[:, 0]
        self.K = lqr_gain(self.Ad, Bd, rho=rho)
        q = [q_x] * 4 + [q_theta] + ([q_gamma] if gamma else [])
        self.Q = np.diag(q)
        self.R = sy ** 2
        self.z0 = np.r_[0, 0, 0, 0, np.log(M.C50N)] if not gamma else np.r_[0, 0, 0, 0, np.log(M.C50N), np.log(M.GAMMA)]
        self.P0 = np.diag([1e-4] * 4 + [p0_theta ** 2] + ([p0_gamma ** 2] if gamma else []))

    def reset(self, pts, plant):
        B = pts["mu"].shape[0]
        self.tgt = pts["tgt"]
        self.z = np.tile(self.z0, (B, 1))
        self.P = np.tile(self.P0, (B, 1, 1))
        self.u_prev = np.zeros(B)

    @staticmethod
    def _sigma(z, P, lam=1.0):
        """(B, 2n+1, n) sigma points and weights"""
        n = z.shape[-1]
        P = 0.5 * (P + np.swapaxes(P, 1, 2))
        A = (n + lam) * P + 1e-12 * np.eye(n)
        try:
            S = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:                           # lost positive definiteness somewhere: floor the spectrum
            w, V = np.linalg.eigh(P)
            S = np.linalg.cholesky((n + lam) * (V * np.maximum(w, 1e-10)[:, None, :]) @ np.swapaxes(V, 1, 2) + 1e-12 * np.eye(n))
        St = np.swapaxes(S, 1, 2)                               # rows = columns of S
        pts = np.concatenate([z[:, None], z[:, None] + St, z[:, None] - St], 1)
        w = np.r_[lam / (n + lam), np.full(2 * n, 1 / (2 * (n + lam)))]
        return pts, w

    def _h(self, s):
        g = np.exp(s[..., 5]) if self.est_gamma else M.GAMMA
        return M.hill(s[..., 3], s[..., 4], g, M.E0, M.EMAX, xp=np)

    def _update(self, z, P, y, pts, w):
        ys = self._h(pts)                                       # (B, 2n+1)
        yp = ys @ w
        dy = ys - yp[:, None]
        Pyy = (dy ** 2) @ w + self.R
        Pzy = np.einsum("j,bjn,bj->bn", w, pts - z[:, None], dy)
        Kg = Pzy / Pyy[:, None]
        return z + Kg * (y - yp)[:, None], P - np.einsum("bi,bj->bij", Kg, Kg) * Pyy[:, None, None]

    def act(self, k, y):
        if k > 0:
            pts, w = self._sigma(self.z, self.P)
            pp = pts.copy()
            pp[..., :4] = pts[..., :4] @ self.Ad.T + self.Bd * self.u_prev[:, None, None]
            zp = np.einsum("j,bjn->bn", w, pp)
            d = pp - zp[:, None]
            Pp = np.einsum("j,bji,bjk->bik", w, d, d) + self.Q
            self.z, self.P = self._update(zp, Pp, y, *self._sigma(zp, Pp))
            self.z[:, :4] = np.maximum(self.z[:, :4], 0)
        else:
            self.z, self.P = self._update(self.z, self.P, y, *self._sigma(self.z, self.P))
            if self.est_gamma:
                self.z[:, :4] = np.maximum(self.z[:, :4], 0)
        g = M.GAMMA
        if self.est_gamma:
            self.z[:, 5] = np.clip(self.z[:, 5], np.log(0.7), np.log(10.0))
            g = np.exp(self.z[:, 5])
        cr = M.cref(np.exp(self.z[:, 4]), g, M.E0, M.EMAX, self.tgt[:, k])
        u = np.clip(M.NOM["Cl1"] * cr - np.sum(self.K * (self.z[:, :4] - cr[:, None]), 1), 0, M.UMAX)
        self.u_prev = u
        return u


class UKFMM(UKFLQR):
    """Observer m assumes the delay dset[m]: at step k it estimates z_j, j = max(0, k - dset[m]) (time update with u_(j-1) whenever j
    advances, measurement update with every reading) and proposes u_m = LQR on its PK estimate predicted over u_j .. u_(k-1). The
    applied input is sum_m p_m u_m, p_m ~ exp(forgetting (lam) sum of log N(innovation_m; 0, Pyy_m)), floored at pfloor. Each observer's
    R follows R <- (1 - beta) R + beta max(nu^2 - (Pyy - R), rmin) (adapt=False: R = sy^2 fixed). With dset=(0,) and adapt=False this is
    UKFLQR (selftest.py checks it). Defaults: the paper's tuning (q_theta, rho, lam, beta on separate patients)."""

    def __init__(self, dset=(0, 2, 4, 6, 8, 10, 12), adapt=True, q_theta=5e-4, rho=2e-4, lam=0.995, beta=0.1, rmin=1.0, pfloor=1e-3, **kw):
        super().__init__(q_theta=q_theta, rho=rho, **kw)
        self.dset, self.adapt, self.lam, self.beta, self.rmin, self.pfloor = np.asarray(dset), adapt, lam, beta, rmin, pfloor

    def reset(self, pts, plant):
        B, Mo = pts["mu"].shape[0], len(self.dset)
        self.tgt, self.B = pts["tgt"], B
        self.z = np.tile(self.z0, (B * Mo, 1))                   # (B * Mo, n): patient-major
        self.P = np.tile(self.P0, (B * Mo, 1, 1))
        self.Rm = np.full(B * Mo, self.R)
        self.ll = np.zeros((B, Mo))
        self.uh = np.zeros((B, M.N))
        self.d = np.tile(self.dset, B)

    def act(self, k, y):
        B, Mo, n = self.B, len(self.dset), self.z.shape[1]
        j = np.maximum(0, k - self.d)                             # the step each observer's state refers to
        adv = k - self.d >= 1                                     # j advanced at this step (by one, once k > d)
        up = np.repeat(self.uh, Mo, 0)[np.arange(B * Mo), np.maximum(j - 1, 0)]
        pts, w = self._sigma(self.z, self.P)
        pp = pts.copy()
        pp[..., :4] = pts[..., :4] @ self.Ad.T + self.Bd * up[:, None, None]
        zp = np.einsum("j,bjn->bn", w, pp)
        dd = pp - zp[:, None]
        Pp = np.einsum("j,bji,bjk->bik", w, dd, dd) + self.Q
        zp, Pp = np.where(adv[:, None], zp, self.z), np.where(adv[:, None, None], Pp, self.P)
        pts2, _ = self._sigma(zp, Pp)
        pts = np.where(adv[:, None, None], pts2, pts)
        ys = self._h(pts)
        yp = ys @ w
        dy = ys - yp[:, None]
        R = self.Rm
        Pyy = (dy ** 2) @ w + R
        Kg = np.einsum("j,bjn,bj->bn", w, pts - zp[:, None], dy) / Pyy[:, None]
        nu = np.repeat(y, Mo) - yp
        self.ll = self.lam * self.ll - 0.5 * (np.log(2 * np.pi * Pyy) + nu ** 2 / Pyy).reshape(B, Mo)
        if self.adapt:
            self.Rm = (1 - self.beta) * R + self.beta * np.maximum(nu ** 2 - (Pyy - R), self.rmin)
        self.z, self.P = zp + Kg * nu[:, None], Pp - np.einsum("bi,bj->bij", Kg, Kg) * Pyy[:, None, None]
        if k > 0 or self.est_gamma:
            self.z[:, :4] = np.maximum(self.z[:, :4], 0)
        g = M.GAMMA
        if self.est_gamma:
            self.z[:, 5] = np.clip(self.z[:, 5], np.log(0.7), np.log(10.0))
            g = np.exp(self.z[:, 5])
        cr = M.cref(np.exp(self.z[:, 4]), g, M.E0, M.EMAX, np.repeat(self.tgt[:, k], Mo))
        xf = self.z[:, :4].copy()                                 # Smith predictor: the PK part forward over u_j .. u_(k-1)
        uhm = np.repeat(self.uh, Mo, 0)
        for s in range(int(self.dset.max())):
            t = j + s
            go = t < k
            xn = xf @ self.Ad.T + self.Bd * uhm[np.arange(B * Mo), np.minimum(t, M.N - 1)][:, None]
            xf = np.where(go[:, None], xn, xf)
        um = (M.NOM["Cl1"] * cr - np.sum(self.K * (xf - cr[:, None]), 1)).reshape(B, Mo)
        p = np.exp(self.ll - self.ll.max(1, keepdims=True))
        p = np.maximum(p / p.sum(1, keepdims=True), self.pfloor)
        p = p / p.sum(1, keepdims=True)
        u = np.clip(np.sum(p * um, 1), 0, M.UMAX)
        self.uh[:, k] = u
        return u


BASELINES = {
    "PID": PID,
    "UKF+LQR": UKFLQR,
    "UKF+LQR+gamma": lambda: UKFLQR(gamma=True),
    "UKF-MM": UKFMM,
    "oracle-CE": OracleCE,
}


def tune(n=300):
    """UKF-MM's q_theta, rho, lam, beta on patients of their own (python baselines.py tune): grid, score = model.score over the validation
    scenarios (random stimulus). The defaults of UKFMM are the result."""
    import itertools, jax
    key = jax.random.PRNGKey(7_000_000)
    pts = {s: M.to_numpy(M.sample_patients(jax.random.fold_in(key, i), n, sp)) for i, (s, sp) in enumerate(M.val_specs().items())}
    rows = []
    for q, rho, lam, beta in itertools.product((2e-4, 5e-4, 1e-3, 2e-3), (1e-4, 2e-4, 5e-4), (0.99, 0.995), (0.03, 0.1)):
        st = {}
        for s, p in pts.items():
            b, u = M.run_numpy(UKFMM(q_theta=q, rho=rho, lam=lam, beta=beta), p)
            m = M.metrics(b, u, p["tgt"])
            st[s] = dict(J=m["J"].mean(), viol=m["viol"].mean())
        rows.append((M.score(st), q, rho, lam, beta))
        print(f"q_theta {q:.0e} rho {rho:.0e} lam {lam} beta {beta}: score {rows[-1][0]:.3f}", flush=True)
    print("best: score %.3f  q_theta %g rho %g lam %g beta %g" % min(rows))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["tune"]:
        tune()
