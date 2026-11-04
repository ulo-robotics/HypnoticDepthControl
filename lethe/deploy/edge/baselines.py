"""
The classical baselines as generated C (CasADi), for the edge benchmarks (edge/baselines, edge.py):
  ukfmm_mpcmm(out)  baselines.c / .h, consts.h:
      ukfmm(z, P, Rm, ll, y, k, tgt, uwin) -> z, P, Rm, ll, u, xf, lc, p     one UKF-MM step (lethe.baselines.classic.UKFMM.estimate + act);
                                                                            uwin[i] = u_(k-13+i), zero before the start
      mpcmm_f / mpcmm_g / mpcmm_h (w, par)                                  MPC-MM's NLP (lethe.baselines.mpc.mpcmm_nlp): cost, gradient,
                                                                            dense Hessian
  mhe_nmpc(out)     mhe_nlp.c / .h, nmpc_nlp.c / .h, mhe_consts.h (lethe.baselines.mpc.MHENMPC with its defaults):
      mhe_f / mhe_g / mhe_h (x, p)    the MHE NLP: cost, gradient, lower triangle of the Hessian (sparse; triplets in mhe_consts.h)
      nmpc_f / nmpc_g / nmpc_h (w, p) the robust NMPC NLP: cost, gradient, dense Hessian
  reference_traces(out, n, names)   trace_<name>.bin: each baseline's closed loop in Python on the patients of lethe.deploy.cpp.write_trace
                                    (mixed_hard, PRNGKey(0)), in its format (int32 n, N; per patient float64 y[N], target[N], u[N])
The generated functions reproduce the Python baselines step by step (the edge drivers check it on every recorded step).
"""
import os, struct, time
import numpy as np

from lethe.core import model as M
from lethe.baselines import classic, mpc

NWIN = 13                                                           # UKF-MM's input window: the longest delay hypothesis + 1


def _generate(out, name, fns):
    import casadi as ca
    cg = ca.CodeGenerator(name + ".c", {"with_header": True})
    for fn in fns:
        cg.add(fn)
    cwd = os.getcwd()
    os.chdir(out)
    try:
        cg.generate()
    finally:
        os.chdir(cwd)


def ukfmm_mpcmm(out):
    import casadi as ca
    os.makedirs(out, exist_ok=True)
    U = classic.UKFMM()
    Ad, Bd, K = U.Ad, U.Bd, U.K
    DS = [int(d) for d in U.dset]
    Mo, n = len(DS), 5
    lamS = 1.0
    w = np.r_[lamS / (n + lamS), np.full(2 * n, 1 / (2 * (n + lamS)))]
    NW = NWIN
    assert max(DS) + 1 <= NW

    def chol(A):
        """lower Cholesky factor of a symmetric positive definite SX matrix (n x n)"""
        L = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = A[i, j] - sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = ca.sqrt(s) if i == j else s / L[j][j]
        return L

    def sigma(z, P):
        P = 0.5 * (P + P.T)
        L = chol((n + lamS) * P + 1e-12 * ca.DM.eye(n))
        cols = [ca.vertcat(*[L[r][c] for r in range(n)]) for c in range(n)]
        return [z] + [z + c for c in cols] + [z - c for c in cols]

    def h(s):
        ce = ca.fmax(s[3], 0.0) ** 2
        return M.E0 - M.EMAX * ce / (ce + ca.exp(s[4]) ** 2)

    def mean(pts):
        return sum(float(wi) * p for wi, p in zip(w, pts))

    z, P = ca.SX.sym("z", n, Mo), ca.SX.sym("P", n * n, Mo)
    Rm, ll = ca.SX.sym("Rm", Mo), ca.SX.sym("ll", Mo)
    y, k, tgt, uw = ca.SX.sym("y"), ca.SX.sym("k"), ca.SX.sym("tgt"), ca.SX.sym("uwin", NW)
    Adm, Bdm = ca.DM(Ad), ca.DM(Bd)
    zo, Po, Ro, lo, xfo = [], [], [], [], []
    for m, d in enumerate(DS):
        zm, Pm = z[:, m], ca.reshape(P[:, m], n, n)
        adv = k - d >= 1
        up = uw[NW - 1 - d]                                         # u_(k-d-1)
        pts = sigma(zm, Pm)
        pp = [ca.vertcat(ca.mtimes(Adm, p[:4]) + Bdm * up, p[4]) for p in pts]
        zp = mean(pp)
        Pp = sum(float(wi) * ca.mtimes(q - zp, (q - zp).T) for wi, q in zip(w, pp)) + ca.DM(U.Q)
        zp, Pp = ca.if_else(adv, zp, zm), ca.if_else(adv, Pp, Pm)
        pts = sigma(zp, Pp)
        ys = [h(p) for p in pts]
        yp = mean(ys)
        R = Rm[m]
        Pyy = sum(float(wi) * (yi - yp) ** 2 for wi, yi in zip(w, ys)) + R
        Kg = sum(float(wi) * (yi - yp) * (p - zp) for wi, yi, p in zip(w, ys, pts)) / Pyy
        nu = y - yp
        lo.append(U.lam * ll[m] - 0.5 * (ca.log(2 * np.pi * Pyy) + nu ** 2 / Pyy))
        Ro.append((1 - U.beta) * R + U.beta * ca.fmax(nu ** 2 - (Pyy - R), U.rmin))
        zn = zp + Kg * nu
        zn = ca.vertcat(ca.if_else(k > 0, ca.fmax(zn[:4], 0), zn[:4]), zn[4])
        zo.append(zn)
        Po.append(ca.vec(Pp - ca.mtimes(Kg, Kg.T) * Pyy))
        # Smith predictor over u_j .. u_(k-1), m = min(k, d) steps: the first d - m of the d slots are skipped
        x = zn[:4]
        mm = ca.fmin(k, d)
        for s in range(d):
            x = ca.if_else(s >= d - mm, ca.mtimes(Adm, x) + Bdm * uw[NW - d + s], x)
        xfo.append(x)
    llv = ca.vertcat(*lo)
    pr = ca.exp(llv - ca.mmax(llv))
    pr = ca.fmax(pr / ca.sum1(pr), U.pfloor)
    pr = pr / ca.sum1(pr)
    lcv = ca.vertcat(*[q[4] for q in zo])
    s_ = (M.E0 - tgt) / M.EMAX
    cr = ca.exp(lcv) * ca.sqrt(s_ / (1 - s_))                       # cref with gamma 2
    um = ca.vertcat(*[M.NOM["Cl1"] * cr[m] - ca.dot(ca.DM(K), xfo[m] - cr[m]) for m in range(Mo)])
    u = ca.fmin(ca.fmax(ca.dot(pr, um), 0), M.UMAX)
    dn = ca.densify
    F = ca.Function("ukfmm", [z, P, Rm, ll, y, k, tgt, uw],
                    [dn(ca.horzcat(*zo)), dn(ca.horzcat(*Po)), dn(ca.vertcat(*Ro)), dn(llv), dn(u),
                     dn(ca.fmax(ca.horzcat(*xfo), 0)), dn(lcv), dn(pr)],
                    ["z", "P", "Rm", "ll", "y", "k", "tgt", "uwin"], ["z1", "P1", "Rm1", "ll1", "u", "xf", "lc", "p"])

    prob, nb = mpc.mpcmm_nlp(Ad, Bd, Mo, mpc.BLOCKS, 1e-3)           # MPCMM's defaults: rho_mpc 1e-3
    wv, par, f = prob["x"], prob["p"], prob["f"]
    H, g = ca.hessian(f, wv)
    Fh = ca.Function("mpcmm_h", [wv, par], [dn(H)])
    _generate(out, "baselines", [F, ca.Function("mpcmm_f", [wv, par], [f]), ca.Function("mpcmm_g", [wv, par], [dn(g)]), Fh])
    z0 = np.tile(U.z0, (Mo, 1)).T
    with open(os.path.join(out, "consts.h"), "w") as fh:
        fh.write(f"#define NOBS {Mo}\n#define NBLK {nb}\n#define NWIN {NW}\n#define UMAX {M.UMAX!r}\n")
        fh.write("static const double Z0[] = {" + ", ".join(repr(float(v)) for v in z0.T.ravel()) + "};\n")
        fh.write("static const double P0[] = {" + ", ".join(repr(float(v)) for v in U.P0.ravel(order="F")) + "};\n")
        fh.write(f"static const double R0 = {float(U.R)!r};\n")
    return f"ukfmm: {F.n_instructions()} instructions; mpcmm Hessian {Fh.n_instructions()}; {nb} move blocks -> {out}/baselines.c"


def mhe_nmpc(out):
    import casadi as ca
    os.makedirs(out, exist_ok=True)
    C = mpc.MHENMPC.__init__.__defaults__
    d = dict(zip(mpc.MHENMPC.__init__.__code__.co_varnames[1:1 + len(C)], C))
    W, Mo = d["W"], len(d["dset"])
    dn = ca.densify
    prob = mpc.mhe_nlp(W, d["q_x"], d["q_lc"], d["s_x0"], d["s_lc0"], d["s_pop"], d["e0_mean"])
    x, p, f = prob["x"], prob["p"], prob["f"]
    H, g = ca.hessian(f, x)
    Hl = ca.tril(H)
    r, c = Hl.sparsity().get_triplet()
    fs = [ca.Function("mhe_f", [x, p], [f]), ca.Function("mhe_g", [x, p], [dn(g)]), ca.Function("mhe_h", [x, p], [Hl])]
    prob2, nb = mpc.nmpc_nlp(Mo, d["blocks"], d["rho"])
    w, p2, f2 = prob2["x"], prob2["p"], prob2["f"]
    H2, g2 = ca.hessian(f2, w)
    fs += [ca.Function("nmpc_f", [w, p2], [f2]), ca.Function("nmpc_g", [w, p2], [dn(g2)]), ca.Function("nmpc_h", [w, p2], [dn(H2)])]
    _generate(out, "mhe_nlp", fs[:3])                               # two files: they compile in parallel
    _generate(out, "nmpc_nlp", fs[3:])

    arr = lambda v: "{" + ", ".join(repr(float(t)) for t in np.ravel(v)) + "}"
    iarr = lambda v: "{" + ", ".join(str(int(t)) for t in np.ravel(v)) + "}"
    with open(os.path.join(out, "mhe_consts.h"), "w") as fh:
        fh.write(f"#define MW {W}\n#define MNOBS {Mo}\n#define MNX {x.shape[0]}\n#define MNP {p.shape[0]}\n#define MNBLK {nb}\n")
        fh.write(f"#define MNNZ {len(r)}\n#define MNP2 {p2.shape[0]}\n#define MDMAX {int(max(d['dset']))}\n")
        fh.write(f"static const int MDSET[] = {iarr(d['dset'])};\nstatic const int MHR[] = {iarr(r)};\nstatic const int MHC[] = {iarr(c)};\n")
        fh.write(f"static const double A3[] = {arr(mpc.A3)};   /* row-major */\nstatic const double B3[] = {arr(mpc.B3)};\n")
        for k, v in (("KE0", mpc.KE0), ("DT", M.DT), ("LC0", mpc.LC0), ("LG0", mpc.LG0), ("SY", d["sy"]), ("KAPPA", d["kappa"]),
                     ("PFLOOR", d["pfloor"]), ("ARR_TAU", d["arr_tau"]), ("E0MEAN", d["e0_mean"]), ("TOL", d["tol"])):
            fh.write(f"static const double {k} = {float(v)!r};\n")
        fh.write(f"static const int MAXIT = {d['max_iter']};\n")
        fh.write(f"static const double SARR0[] = {arr(d['s_arr'])};\nstatic const double SARR1[] = {arr(d['s_arr_end'])};\n")
        fh.write(f"static const double THLB[] = {arr([np.log(0.8), -1.6, 80.0])};\nstatic const double THUB[] = {arr([np.log(8.0), 1.6, 105.0])};\n")
        fh.write(f"static const double LCLB = {float(mpc.LC0 - 2.0)!r}, LCUB = {float(mpc.LC0 + 2.0)!r};\n")
    return (f"mhe: {x.shape[0]} variables, {p.shape[0]} parameters, Hessian {len(r)} nonzeros, {fs[2].n_instructions()} instructions; "
            f"nmpc Hessian {fs[5].n_instructions()} -> {out}/mhe_nlp.c, nmpc_nlp.c")


def reference_traces(out, n=8, names=("ukfmm", "mpcmm", "mhe"), seed=0):
    """closed loops of the Python baselines; -> {name: seconds}"""
    import jax
    os.makedirs(out, exist_ok=True)
    pts = M.to_numpy(M.sample_patients(jax.random.PRNGKey(seed), n, M.VAL["mixed_hard"][0]))
    make = dict(ukfmm=classic.UKFMM, mpcmm=mpc.MPCMM, mhe=mpc.MHENMPC)
    took = {}
    for name in names:
        t0 = time.time()
        ctrl, plant = make[name](), M.Plant(pts)
        ctrl.reset(pts, plant)
        N = plant.N
        ys, us = np.zeros((n, N)), np.zeros((n, N))
        for k in range(N):
            y = plant.measure()
            u = np.clip(np.asarray(ctrl.act(k, y), np.float64), 0.0, M.UMAX)
            ys[:, k], us[:, k] = y, u
            plant.step(u)
        with open(os.path.join(out, f"trace_{name}.bin"), "wb") as f:
            f.write(struct.pack("<2i", n, N))
            for i in range(n):
                for a in (ys[i], pts["tgt"][i], us[i]):
                    f.write(np.ascontiguousarray(a, "<f8").tobytes())
        took[name] = time.time() - t0
    return took
