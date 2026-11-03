#!/usr/bin/env python
"""
Evaluate a trained policy against the baselines on the SAME fresh patients (a key never used in training or validation).

  python evaluate.py runs/train/best_model.pkl                       # validation scenarios, 1000 patients each
  python evaluate.py runs/train/best_model.pkl --stress --n 2000     # + the out-of-distribution stress scenarios
  python evaluate.py runs/train/best_model.pkl --prior --plot        # + the untrained policy, + a figure of BIS / infusion quantiles
  python evaluate.py runs/ft/best_model.pkl --extra winner=runs/search/best_model.pkl --stim-times 10,15,20,25
                                                                   # + another model; + a stimulus of 0.3 at each of these times (min)

The policy runs in the JAX closed loop, the baselines on the numpy plant (both checked equal by selftest.py). Prints, per scenario, the mean
J of every controller and the paired difference policy - baseline (95% CI, share of patients where the policy is better). Writes
<out>/results.json and per-patient metrics <out>/<scenario>.npz.
"""
import argparse, json, os, time
import numpy as np
import jax
import jax.numpy as jnp

import model as M
import policy as P
import baselines as BL

jax.config.update("jax_default_matmul_precision", "highest")    # float32 matmuls (not TF32): the policy computes what it was trained on
EVAL_KEY = jax.random.PRNGKey(8_000_000)


_ROLLOUT = jax.jit(lambda p, pt: P.rollout(p, pt)[1:])


def run_policy(params, pts, chunk=1024):
    f = _ROLLOUT
    B = pts["mu"].shape[0]
    out = [f(params, {k: (v[:, :, i:i + chunk] if k == "z" else v[i:i + chunk]) for k, v in pts.items()}) for i in range(0, B, chunk)]
    b, u = (np.concatenate([np.asarray(o[j], np.float64) for o in out], 1).T for j in (0, 1))
    return b, u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=1000, help="patients per scenario")
    ap.add_argument("--out", default="", help="default: <model folder>/eval")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--prior", action="store_true", help="also the untrained policy (prior compensator) of the model's config")
    ap.add_argument("--extra", action="append", default=[], help="name=path: another model on the same patients")
    ap.add_argument("--stim-times", default="", help="comma separated times (min): a scenario with a stimulus of 0.3 at each")
    ap.add_argument("--baselines", default=",".join(BL.BASELINES), help="comma separated, from: " + ", ".join(BL.BASELINES))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.model)), "eval")
    os.makedirs(out, exist_ok=True)

    m = P.load_model(args.model)
    ctrls = {"policy": jax.tree_util.tree_map(jnp.asarray, m["params"])}               # name -> parameters
    if args.prior:
        cfg = {**P.ARCH_DEFAULTS, **(m.get("cfg") or {})}
        ctrls["prior"] = P.init_params(jax.random.PRNGKey(0), cfg)
    for e in args.extra:
        n, path = e.split("=", 1)
        ctrls[n] = jax.tree_util.tree_map(jnp.asarray, P.load_model(path)["params"])
    names = [b for b in args.baselines.split(",") if b]
    scen = {n: sp for n, (sp, _w) in M.VAL.items()}
    if args.stress:
        scen.update(M.STRESS)
    for t in (float(x) for x in args.stim_times.split(",") if x):
        scen[f"stim_at_{t:g}"] = M.spec(stim_p=1.0, stim_lo=0.3, stim_hi=0.3, stim_tlo=t, stim_thi=t)

    res, curves = {}, {}
    for si, (sname, sp) in enumerate(scen.items()):
        t0 = time.time()
        pts = M.sample_patients(jax.random.fold_in(EVAL_KEY, si), args.n, sp)
        pn = M.to_numpy(pts)
        per, curves[sname] = {}, {}
        for cname, prm in ctrls.items():
            b, u = run_policy(prm, pts)
            per[cname] = M.metrics(b, u, pn["tgt"])
            curves[sname][cname] = (b, u)
        for cname in names:
            b, u = M.run_numpy(BL.BASELINES[cname](), pn)
            per[cname] = M.metrics(b, u, pn["tgt"])
            curves[sname][cname] = (b, u)
        np.savez(os.path.join(out, f"{sname}.npz"), **{f"{c}__{k}": v for c, d in per.items() for k, v in d.items()})
        res[sname] = {c: {k: float(np.nanmean(v)) for k, v in d.items()} for c, d in per.items()}
        J = per["policy"]["J"]
        print(f"\n== {sname}  ({args.n} patients, {time.time() - t0:.0f} s)\n{'controller':16s} {'J':>8s} {'J ind':>8s} {'J maint':>8s} {'%<40':>6s}"
              f" {'policy - it (95% CI)':>22s} {'win':>5s}")
        for c, d in per.items():
            row = f"{c:16s} {np.mean(d['J']):8.2f} {np.mean(d['J_ind']):8.2f} {np.mean(d['Jm']):8.2f} {np.mean(d['low']):6.2f}"
            if c != "policy":
                diff = J - d["J"]
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                res[sname][c].update(diff=float(diff.mean()), diff_se=float(se), win=float(np.mean(diff < 0)))
                row += f" {diff.mean():+9.2f} +- {1.96 * se:5.2f}      {100 * np.mean(diff < 0):4.0f}%"
            print(row, flush=True)

    val = [n for n in M.VAL if n in res]
    print("\nscore (weighted mean over the validation scenarios of J + 2 viol):")
    for c in res[val[0]]:
        print(f"  {c:16s} {M.score({n: res[n][c] for n in val}):8.3f}")
        for n in val:
            res[n][c]["score_part"] = res[n][c]["J"] + 2.0 * res[n][c]["viol"]
    json.dump(dict(model=os.path.abspath(args.model), n=args.n, results=res), open(os.path.join(out, "results.json"), "w"), indent=1)
    print(f"\nwritten to {out}")
    if args.plot:
        plot(curves, os.path.join(out, "traces.png"))


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [s for s in ("nominal", "delay_60s", "mixed_hard") if s in curves]
    ctr = [c for c in ("policy", "UKF-MM", "UKF+LQR", "oracle-CE") if c in curves[show[0]]]
    t = np.arange(M.N) * M.DT
    fig, ax = plt.subplots(2, len(show), figsize=(5 * len(show), 6), sharex=True, squeeze=False)
    for j, s in enumerate(show):
        for c in ctr:
            b, u = curves[s][c]
            for i, (a, lo_hi) in enumerate(((b, (25, 75)), (u, (25, 75)))):
                q = np.percentile(a, [lo_hi[0], 50, lo_hi[1]], 0)
                ax[i, j].plot(t, q[1], label=c)
                ax[i, j].fill_between(t, q[0], q[2], alpha=0.2)
        ax[0, j].set_title(s); ax[0, j].axhline(40, color="k", lw=0.5, ls=":")
        ax[0, j].set_ylabel("BIS (median, IQR)"); ax[1, j].set_ylabel("infusion mg/min"); ax[1, j].set_xlabel("min")
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
