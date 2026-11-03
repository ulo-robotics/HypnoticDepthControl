"""
Model-predictive baselines (CasADi + IPOPT; optional dependency, imported when a controller is built).

MPC-MM: UKF-MM's delay-aware multiple-model estimator (classic.UKFMM.estimate) with a robust nonlinear MPC in place of its LQR mixture. Each
step, per patient, one NLP over the infusion plan (held over move blocks, 0 <= u <= UMAX) minimizes the probability-weighted score cost over
the observers' predictions: sum_m p_m mean_i [(BIS_mi - target)^2 + safe_w softplus_beta(40 - BIS_mi)^2] + rho (squared input moves),
BIS_mi from observer m's Smith-predicted PK state with the nominal PK and Hill shape and its C50 estimate (held over the horizon). The
current target only (no preview).

MHE+NMPC: full-state moving-horizon estimation with process noise and a random-walk C50, then the same robust NMPC. Model: the 4-state PK
(three compartments with the nominal parameters, the effect site with rate s * ke0_nom), BIS = Hill(ce; C50, gamma) with E0 estimated and
the nominal Emax, the monitor d steps late. Unknowns: the state trajectory and theta = (log gamma, log s, E0), per delay hypothesis d (the
bank 0, 2, .., 12 steps, as UKF-MM's). The MHE, every step, per patient and hypothesis, over the model times t0 .. t0 + W (t0 = k - d - W;
reading k - W + i shows BIS at t0 + i):
    sum_i (y - Hill(ce_i))^2 / sy^2                                    measurements
  + sum_i |x_{i+1} - f_theta(x_i, u_i)|^2 / q_x + (lc_{i+1} - lc_i)^2 / q_lc      process noise, C50 random walk
  + |x_0 - xbar_0|^2 / P0 + |theta - theta_prev|^2 / s_arr^2 + |(lc_0, theta) - population|^2 / s_pop^2     arrival and population priors
with x >= 0; xbar_0 is the previous window's estimate one step further. s_arr decays from s_arr to s_arr_end with time constant arr_tau
steps (free while the data are informative, steadier against the noise later). The estimate at model time k - d is propagated to now over
the known infusions (the delay's Smith step). Probabilities p_d ~ exp(-kappa (J_d - min J)), floored. The defaults are the tuned ones.

All patients of a batch go to IPOPT in one call (Function.map); CasADi's thread map serializes IPOPT, so batches run in parallel processes
(baselines/parallel.py) with threads=1.
"""
import hashlib, os, subprocess
from functools import partial
import numpy as np

from lethe.core import model as M
from lethe.baselines import classic

BLOCKS = (1, 2, 3, 6, 12, 36)          # move blocks: 60 steps = 5 min
CODEGEN_DIR = os.path.expanduser("~/.cache/lethe/casadi")
LC0, LG0 = np.log(M.C50N), np.log(M.GAMMA)
KE0 = M.NOM["ke0"]
_Ad, _Bd = M.discretize(*M.pk_matrices(M.NOM))
A3, B3 = _Ad[:3, :3], _Bd[:3, 0]
OPTS = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}


def _ca():
    import casadi
    return casadi


def softplus_low(b, beta):
    """smooth max(40 - BIS, 0)"""
    return _ca().log1p(_ca().exp(beta * (40.0 - b))) / beta


# ---------------------------------------------------------------- MPC-MM
def mpcmm_nlp(Ad, Bd, n_obs, blocks=BLOCKS, rho=0.0, safe_w=2.0, beta=2.0, gamma=M.GAMMA, e0=M.E0, emax=M.EMAX):
    """x = u (n_blocks); p = [x0 (n_obs x 4, observer-major); log C50 (n_obs); probabilities (n_obs); target; u_prev]"""
    ca = _ca()
    nb, Np = len(blocks), sum(blocks)
    u, p = ca.SX.sym("u", nb), ca.SX.sym("p", 6 * n_obs + 2)
    tgt, up = p[6 * n_obs], p[6 * n_obs + 1]
    A, Bv = ca.DM(Ad), ca.DM(Bd)
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
                Jm += (b - tgt) ** 2 + safe_w * softplus_low(b, beta) ** 2
        J += pm * Jm / Np
    return dict(x=u, p=p, f=J), nb


class MPCMM(classic.UKFMM):
    def __init__(self, blocks=BLOCKS, rho_mpc=1e-3, threads=1, tol=1e-6, map_only=False, **kw):
        """map_only: plan on the most likely observer alone"""
        super().__init__(**kw)
        ca = _ca()
        self.blocks, self.threads, self.map_only = blocks, threads, map_only
        prob, self.nb = mpcmm_nlp(self.Ad, self.Bd, len(self.dset), blocks, rho_mpc)
        self.S = ca.nlpsol("mpcmm", "ipopt", prob, dict(OPTS, **{"ipopt.tol": tol, "ipopt.max_iter": 100, "ipopt.mu_strategy": "adaptive"}))
        self._maps = {}

    def reset(self, pts, plant):
        super().reset(pts, plant)
        B = self.B
        if B not in self._maps:
            self._maps[B] = self.S.map(B, "thread", self.threads)
        self.F = self._maps[B]
        self.w = np.zeros((self.nb, B))
        self.u_prev = np.zeros(B)
        self.lbx, self.ubx = np.zeros((self.nb, B)), np.full((self.nb, B), M.UMAX)

    def act(self, k, y):
        B, Mo = self.B, len(self.dset)
        xf, lc, _, p = self.estimate(k, y)
        xf = np.maximum(xf, 0.0)
        if self.map_only:
            p = np.eye(Mo)[np.argmax(p, 1)]
        P = np.concatenate([xf.reshape(B, Mo * 4), lc.reshape(B, Mo), p, self.tgt[:, k][:, None], self.u_prev[:, None]], 1).T
        w0 = np.concatenate([self.w[1:], self.w[-1:]], 0)
        r = self.F(x0=self.w, p=P, lbx=self.lbx, ubx=self.ubx)               # warm start: the previous plan
        w = np.array(r["x"])
        self.w = np.where(np.isfinite(w), w, w0)
        u = np.clip(self.w[0], 0.0, M.UMAX)
        self.u_prev = u
        self.uh[:, k] = u
        return u


# ---------------------------------------------------------------- MHE + NMPC
def _hill(ce, lc, lg, e0=M.E0, emax=M.EMAX):
    ca = _ca()
    g = ca.exp(lg)
    ceg = ca.fmax(ce, 1e-12) ** g
    return e0 - emax * ceg / (ceg + ca.exp(g * lc))


def _f(x, u, a):
    """one step: nominal compartments (exact ZOH), effect site with factor a = exp(-s ke0 DT) on the step's mean central concentration"""
    ca = _ca()
    c = ca.mtimes(ca.DM(A3), x[:3]) + ca.DM(B3) * u
    ce = a * x[3] + (1 - a) * 0.5 * (x[0] + c[0])
    return ca.vertcat(c, ce)


def f_np(x, u, a):
    c = x[..., :3] @ A3.T + B3 * u[..., None]
    ce = a * x[..., 3] + (1 - a) * 0.5 * (x[..., 0] + c[..., 0])
    return np.concatenate([c, ce[..., None]], -1)


def mhe_nlp(W, q_x, q_lc, s_x0, s_lc0, s_pop, e0_mean=97.0):
    """x = [X (5 x (W + 1), column-major: the 4 PK states and log C50); theta = (log gamma, log s, E0)]; p = [xbar_0 (5); u (W); y (W);
    mask (W); theta_prev (3); arrival stds of theta (3); measurement noise sd]"""
    ca = _ca()
    X, th = ca.SX.sym("X", 5, W + 1), ca.SX.sym("th", 3)
    p = ca.SX.sym("p", 5 + 3 * W + 7)
    xb, u, y, mk = p[0:5], p[5:5 + W], p[5 + W:5 + 2 * W], p[5 + 2 * W:5 + 3 * W]
    prev, s_arr, sy = p[5 + 3 * W:8 + 3 * W], p[8 + 3 * W:11 + 3 * W], p[11 + 3 * W]
    a = ca.exp(-ca.exp(th[1]) * KE0 * M.DT)
    J = 0
    for i in range(W):
        r = X[:4, i + 1] - _f(X[:4, i], u[i], a)
        J += ca.sumsqr(r) / q_x + (X[4, i + 1] - X[4, i]) ** 2 / q_lc
        J += mk[i] * (y[i] - _hill(X[3, i + 1], X[4, i + 1], th[0], e0=th[2])) ** 2 / sy ** 2
    J += ca.sumsqr((X[:4, 0] - xb[:4]) / (s_x0 * ca.fmax(xb[:4], 0.05))) + ((X[4, 0] - xb[4]) / s_lc0) ** 2
    J += ca.sumsqr((th - prev) / s_arr) + ca.sumsqr((ca.vertcat(X[4, 0], th) - ca.vertcat(LC0, LG0, 0.0, e0_mean)) / ca.DM(s_pop))
    return dict(x=ca.vertcat(ca.vec(X), th), p=p, f=J)


def nmpc_nlp(n_obs, blocks, rho, safe_w=2.0, beta=2.0):
    """x = u blocks; p = [per hypothesis: x now (4), log C50, log gamma, a, probability, E0 (9 n_obs); target; u_prev]"""
    ca = _ca()
    nb, Np = len(blocks), sum(blocks)
    u, p = ca.SX.sym("u", nb), ca.SX.sym("p", 9 * n_obs + 2)
    tgt, up = p[9 * n_obs], p[9 * n_obs + 1]
    useq = [u[j] for j, L in enumerate(blocks) for _ in range(L)]
    J = rho * ((u[0] - up) ** 2 + ca.sumsqr(u[1:] - u[:-1]))
    for m in range(n_obs):
        q = p[9 * m: 9 * m + 9]
        x, lc, lg, a, pm, e0 = q[0:4], q[4], q[5], q[6], q[7], q[8]
        Jm = 0
        for i in range(Np):
            x = _f(x, useq[i], a)
            b = _hill(x[3], lc, lg, e0=e0)
            Jm += (b - tgt) ** 2 + safe_w * softplus_low(b, beta) ** 2
        J += pm * Jm / Np
    return dict(x=u, p=p, f=J), nb


_SOLVERS = {}


def _solver(name, prob, opts, codegen):
    """IPOPT on the NLP; codegen: its functions generated as C and compiled once (gcc -O3, cached by content in CODEGEN_DIR) instead of
    CasADi's virtual machine"""
    ca = _ca()
    S = ca.nlpsol(name, "ipopt", prob, opts)
    if not codegen:
        return S
    os.makedirs(CODEGEN_DIR, exist_ok=True)
    tag = hashlib.sha1((str(prob["f"]) + str(prob["x"].shape) + str(prob["p"].shape)).encode()).hexdigest()[:12]
    so = os.path.join(CODEGEN_DIR, f"{name}_{tag}.so")
    if so not in _SOLVERS:
        if not os.path.exists(so):
            src = os.path.join(CODEGEN_DIR, f"{name}_{tag}.c")
            cwd = os.getcwd()
            os.chdir(CODEGEN_DIR)
            try:
                S.generate_dependencies(os.path.basename(src))
                subprocess.run(["gcc", "-fPIC", "-shared", "-O3", src, "-o", so + ".tmp"], check=True)
                os.replace(so + ".tmp", so)
            finally:
                os.chdir(cwd)
        _SOLVERS[so] = ca.nlpsol(name, "ipopt", so, opts)
    return _SOLVERS[so]


class MHENMPC:
    def __init__(self, dset=(0, 2, 4, 6, 8, 10, 12), W=36, sy=M.SY, q_x=1e-4, q_lc=2e-3, s_x0=0.2, s_lc0=0.05, s_arr=(0.005, 0.005, 2.0),
                 s_arr_end=(0.03, 0.03, 0.05), arr_tau=12.0, s_pop=(0.4, 0.4, 0.5, 4.0), kappa=1.0, e0_mean=97.0, pfloor=1e-3,
                 blocks=BLOCKS, rho=1e-3, threads=1, tol=1e-6, max_iter=60, codegen=False):
        self.dset, self.W, self.kappa, self.pfloor, self.threads, self.sy = np.asarray(dset), W, kappa, pfloor, threads, sy
        self.s_arr0, self.s_arr1, self.arr_tau = np.asarray(s_arr, float), np.asarray(s_arr_end, float), arr_tau
        opts = dict(OPTS, **{"ipopt.tol": tol, "ipopt.max_iter": max_iter})
        self.Smhe = _solver("mhe", mhe_nlp(W, q_x, q_lc, s_x0, s_lc0, s_pop, e0_mean), opts, codegen)
        self.e0_mean = e0_mean
        prob, self.nb = nmpc_nlp(len(dset), blocks, rho)
        self.Smpc = _solver("mpc", prob, dict(opts, **{"ipopt.mu_strategy": "adaptive"}), codegen)
        self.nx = 5 * (W + 1)
        self.thlb = np.array([np.log(0.8), -1.6, 80.0])
        self.thub = np.array([np.log(8.0), 1.6, 105.0])
        self.lclb, self.lcub = LC0 - 2.0, LC0 + 2.0
        self._maps = {}

    def _map(self, S, n):
        key = (id(S), n)
        if key not in self._maps:
            self._maps[key] = S.map(n, "thread", self.threads)
        return self._maps[key]

    def reset(self, pts, plant):
        B, Mo, W, N = pts["mu"].shape[0], len(self.dset), self.W, pts["z"].shape[0]
        self.B, self.N, self.tgt = B, N, pts["tgt"]
        self.Fm, self.Fc = self._map(self.Smhe, B * Mo), self._map(self.Smpc, B)
        self.th = np.tile(np.array([LG0, 0.0, self.e0_mean]), (B * Mo, 1))
        self.X = np.zeros((B * Mo, W + 1, 5))                     # the last window's estimate (PK states, log C50), model times t0 .. t0 + W
        self.X[..., 4] = LC0
        self.yh = np.zeros((B, N))
        self.uh = np.zeros((B, N))
        self.w = np.zeros((self.nb, B))
        self.u_prev = np.zeros(B)
        self.d = np.tile(self.dset, B)
        self.row = np.repeat(np.arange(B), Mo)

    def act(self, k, y):
        B, Mo, W = self.B, len(self.dset), self.W
        n = B * Mo
        self.yh[:, k] = y
        t0 = k - self.d - W                                                # model time of the window start, per hypothesis
        mt = t0[:, None] + np.arange(W + 1)[None, :]                       # (n, W + 1) model times of X
        rt = k - W + np.arange(1, W + 1)                                   # reading times of y_1 .. y_W
        a = np.exp(-np.exp(self.th[:, 1]) * KE0 * M.DT)
        xb = self.X[:, 1].copy() if k > 0 else np.concatenate([np.zeros((n, 4)), np.full((n, 1), LC0)], 1)   # arrival: previous X_1
        xb[:, :4] = np.where((mt[:, 0] >= 0)[:, None], np.maximum(xb[:, :4], 0.0), 0.0)
        ut = mt[:, :W]
        uu = np.where(ut >= 0, self.uh[self.row[:, None], np.clip(ut, 0, None)], 0.0)
        yy = np.where(rt[None, :] >= 0, self.yh[self.row[:, None], np.clip(rt, 0, None)[None, :]], 0.0)
        mk = np.broadcast_to((rt >= 0).astype(float), (n, W))
        sa = self.s_arr1 + (self.s_arr0 - self.s_arr1) * np.exp(-k / self.arr_tau)
        P = np.concatenate([xb, uu, yy, mk, self.th, np.tile(sa, (n, 1)), np.full((n, 1), self.sy)], 1).T
        if k > 0:                                                          # warm start: the previous window shifted, its last state predicted
            last = np.concatenate([f_np(self.X[:, -1, :4], uu[:, -1], a), self.X[:, -1, 4:]], 1)
            Xw = np.concatenate([self.X[:, 1:], last[:, None]], 1)
        else:
            Xw = self.X.copy()
        lbX = np.zeros((n, W + 1, 5))
        lbX[..., 4] = self.lclb
        ubX = np.concatenate([np.where((mt >= 0)[..., None], 1e3, 0.0) * np.ones((1, 1, 4)), np.full((n, W + 1, 1), self.lcub)], -1)
        Xw = np.clip(Xw, lbX, ubX)
        x0 = np.concatenate([Xw.reshape(n, -1), self.th], 1).T
        lb = np.concatenate([lbX.reshape(n, -1), np.tile(self.thlb, (n, 1))], 1).T
        ub = np.concatenate([ubX.reshape(n, -1), np.tile(self.thub, (n, 1))], 1).T
        r = self.Fm(x0=x0, p=P, lbx=lb, ubx=ub)
        sol = np.array(r["x"]).T
        ok = np.isfinite(sol).all(1)
        self.X[ok] = sol[ok, :self.nx].reshape(-1, W + 1, 5)                 # ca.vec is column-major: (W + 1) columns of 5
        self.th[ok] = sol[ok, self.nx:]
        Jw = np.array(r["f"]).reshape(B, Mo)
        p = np.exp(-self.kappa * (Jw - Jw.min(1, keepdims=True)))
        p = np.maximum(p / p.sum(1, keepdims=True), self.pfloor)
        p = p / p.sum(1, keepdims=True)
        # present state: the window's last estimate (model time k - d) over the known infusions u_{k-d} .. u_{k-1}
        a = np.exp(-np.exp(self.th[:, 1]) * KE0 * M.DT)
        xn = self.X[:, -1, :4].copy()
        for s in range(int(self.dset.max())):
            t = k - self.d + s
            go = (t < k) & (t >= 0)
            un = self.uh[self.row, np.clip(t, 0, self.N - 1)]
            xn = np.where(go[:, None], f_np(xn, un, a), xn)
        lc = self.X[:, -1, 4].reshape(B, Mo, 1)                            # the random walk: the current log C50 is the window's last
        per = np.concatenate([xn.reshape(B, Mo, 4), lc, self.th[:, :1].reshape(B, Mo, 1), a.reshape(B, Mo, 1), p[..., None],
                              self.th[:, 2].reshape(B, Mo, 1)], -1).reshape(B, 9 * Mo)
        Pc = np.concatenate([per, self.tgt[:, k][:, None], self.u_prev[:, None]], 1).T
        rc = self.Fc(x0=self.w, p=Pc, lbx=np.zeros((self.nb, B)), ubx=np.full((self.nb, B), M.UMAX))
        w = np.array(rc["x"])
        self.w = np.where(np.isfinite(w), w, self.w)
        u = np.clip(self.w[0], 0.0, M.UMAX)
        self.u_prev = u
        self.uh[:, k] = u
        return u


# name -> controller factory (picklable: the parallel runner builds one per worker process)
BASELINES = {"MPC-MM": partial(MPCMM), "MHE+NMPC": partial(MHENMPC)}
