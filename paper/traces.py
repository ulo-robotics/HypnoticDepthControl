"""BIS / infusion quantiles for the paper figures: re-runs evaluate.py closed loops (same key and scenario order, the same patients) of a
few scenarios and checks the recomputed per-patient J against the stored one; and the cost per 30-min segment over 250 min.

  python paper/traces.py [RUN_DIR] [OLD_LONG]

RUN_DIR (default runs/finetune) holds best_model.pkl and the evaluations eval30/ and eval250/; writes RUN_DIR/eval{30,250}/traces_q.npz
and RUN_DIR/paper/long_segments.json. OLD_LONG (optional): the per-segment costs of the earlier model with the elapsed-time input
(old_long.json: 300 patients per scenario, PRNGKey(77 + i), its UKF-MM segments the check that the patients are the same); without it
the figure has no dashed curve.
"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp
from lethe.core import model as M
from lethe.core import policy as P
import lethe.baselines as BL
from lethe.evaluate import EVAL_KEY, run_policy

R = sys.argv[1] if len(sys.argv) > 1 else "runs/finetune"
OLD = sys.argv[2] if len(sys.argv) > 2 else None
pol = lambda p: jax.tree_util.tree_map(jnp.asarray, P.load_model(p)["params"])
ctrls = {"policy": pol(f"{R}/best_model.pkl")}
jobs = [  # (eval folder, minutes, n, with stress + stimulus times, scenarios, baselines)
    ("eval30", 30.0, 2000, True, ["nominal", "delay_60s", "mixed_hard"], ["UKF-MM", "UKF+LQR"]),
    ("eval250", 250.0, 500, False, ["nominal", "stimulus", "mixed_hard"], ["UKF-MM", "oracle-CE"]),
]
for folder, minutes, n, stress, want, base in jobs:
    scen = {k: sp for k, (sp, _w) in M.VAL.items()}
    if stress:
        scen.update(M.STRESS)
        for t in (10.0, 15.0, 20.0, 25.0):
            scen[f"stim_at_{t:g}"] = M.spec(stim_p=1.0, stim_lo=0.3, stim_hi=0.3, stim_tlo=t, stim_thi=t)
    scen = {k: dict(sp, T=minutes) for k, sp in scen.items()}
    out = {}
    for si, (s, sp) in enumerate(scen.items()):
        if s not in want:
            continue
        pts = M.sample_patients(jax.random.fold_in(EVAL_KEY, si), n, sp)
        pn = M.to_numpy(pts)
        ref = np.load(f"{R}/{folder}/{s}.npz")
        for c in list(ctrls) + base:
            b, u = run_policy(ctrls[c], pts) if c in ctrls else BL.run(c, pn)
            J = M.metrics(b, u, pn["tgt"])["J"]
            print(f"{folder} {s:12s} {c:10s} max |J - stored J| = {np.max(np.abs(J - ref[c + '__J'])):.2e}", flush=True)
            for nm, a in (("b", b), ("u", u)):
                out[f"{s}__{c}__{nm}"] = np.percentile(a, [10, 25, 50, 75, 90], 0).astype(np.float32)
    np.savez(f"{R}/{folder}/traces_q.npz", **out)
    print("written", folder, flush=True)

# cost J + 2 viol per 30-min segment over 250 min, 300 patients per scenario; with OLD_LONG, the same patients as the earlier model's
# and UKF-MM recomputed as the check that they are the same
old = json.load(open(OLD)) if OLD else None
seg = {}
for i, nm in enumerate(("nominal", "stimulus", "mixed_hard")):
    sp = M.with_stimulus(M.VAL[nm][0], "signed") if nm == "stimulus" else M.VAL[nm][0]
    pts = M.sample_patients(jax.random.PRNGKey(77 + i), 300, dict(sp, T=250.0))
    pn = M.to_numpy(pts)
    seg[nm] = {"old": old[nm]["old"]} if old else {}
    for c in ("policy", "UKF-MM", "oracle-CE"):
        b, u = run_policy(ctrls[c], pts) if c in ctrls else BL.run(c, pn)
        e = (b - pn["tgt"]) ** 2 + 2 * np.maximum(40 - b, 0) ** 2
        seg[nm][c] = dict(segs=[float(e[:, s * 360:(s + 1) * 360].mean()) for s in range(b.shape[1] // 360)], low=float(100 * np.mean(b < 40)))
    if old:
        d = np.max(np.abs(np.array(seg[nm]["UKF-MM"]["segs"]) - np.array(old[nm]["UKF-MM"]["segs"])))
        print(f"long {nm}: UKF-MM segments vs {OLD} max diff {d:.2e}", flush=True)
    print(f"long {nm}: policy {np.round(seg[nm]['policy']['segs'], 1)}", flush=True)
os.makedirs(f"{R}/paper", exist_ok=True)
json.dump(seg, open(f"{R}/paper/long_segments.json", "w"), indent=1)
print("written long_segments.json")
