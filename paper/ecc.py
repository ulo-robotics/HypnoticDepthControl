"""Tables and figure of the 6-page ECC paper, from the same stored results as analyze.py.

  python ecc.py DATA_DIR ECC_DIR

Writes ECC_DIR/tab_main.tex, tab_ladder.tex, tab_long.tex, tab_edge.tex, fig/long_col.pdf and ECC_DIR/numbers_ecc.json. A result whose
data folder is missing is written in red through \\pend{...} (defined in main.tex) as a dash.
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "fig"), exist_ok=True)
VAL = ["nominal", "stimulus", "fast_drift", "delay_30s", "delay_60s", "gamma_wide", "pk_shift", "target_range", "target_steps", "noise_steps",
       "mixed_hard"]
W = np.array([2.0] + [1.0] * 9 + [2.0])
NICE = {"nominal": "nominal", "stimulus": "stimulus", "fast_drift": "fast drift", "delay_30s": "delay 30 s", "delay_60s": "delay 60 s",
        "gamma_wide": r"wide $\gamma$", "pk_shift": "PK shift", "target_range": "target range", "target_steps": "target steps",
        "noise_steps": "noise steps", "mixed_hard": "mixed (hard)"}
nums = {}


def load(folder):
    E = {}
    for s in VAL:
        p = os.path.join(DATA, folder, f"{s}.npz")
        if os.path.exists(p):
            z = np.load(p)
            for k in z.files:
                c, m = k.split("__")
                E.setdefault(c, {}).setdefault(s, {})[m] = z[k]
    return E


def merge(E, folder):
    """the MPC baselines of a separate run on the same patients (checked: Lethe agrees per patient)"""
    Em = load(folder)
    for s in VAL:
        if s in Em.get("policy", {}):
            assert np.max(np.abs(Em["policy"][s]["J"] - E["policy"][s]["J"])) < 1e-3, (folder, s)
            for c in ("MPC-MM", "MHE+NMPC"):
                E.setdefault(c, {})[s] = Em[c][s]


def paired(a, b):
    d = a - b
    return float(d.mean()), float(1.96 * d.std(ddof=1) / math.sqrt(len(d))), float((d < 0).mean())


have = lambda E, c: c in E and all(s in E[c] for s in VAL)
score_pp = lambda E, c: sum(w * (E[c][s]["J"] + 2 * E[c][s]["viol"]) for w, s in zip(W, VAL)) / W.sum()
f1 = lambda x, n=1: f"{x:.{n}f}"
sg = lambda x, n=1: f"{x:+.{n}f}".replace("-", "$-$")
pend = lambda t: r"\pend{" + t + "}"


def table(E, cols, names, metric, fname, ref="UKF-MM", dg=1):
    """mean metric per controller and validation scenario, Lethe - ref (95% CI) and win share, time below 40; last row the score S"""
    causal = [c for c in cols if c != "oracle-CE" and have(E, c)]
    L = [r"\begin{tabular}{l" + "r" * len(cols) + "rr}", r"\toprule",
         "scenario & " + " & ".join(names[c] if have(E, c) else pend(names[c]) for c in cols) + rf" & Lethe$-${names[ref]} & win\\", r"\midrule"]
    for s in VAL:
        v = {c: float(E[c][s][metric].mean()) for c in cols if have(E, c)}
        best = min(causal, key=lambda c: v[c])
        cells = [(r"\textbf{" + f1(v[c], dg) + "}" if c == best else f1(v[c], dg)) if c in v else pend("--") for c in cols]
        d, ci, win = paired(E["policy"][s][metric], E[ref][s][metric])
        L.append(f"{NICE[s]} & " + " & ".join(cells) + f" & {sg(d, dg)}$\\pm${f1(ci, dg)} & {100 * win:.0f}\\%\\\\")
    L.append(r"\midrule")
    sc = {c: float(score_pp(E, c).mean()) for c in cols if have(E, c)}
    best = min(causal, key=lambda c: sc[c])
    cells = [(r"\textbf{" + f1(sc[c], 2) + "}" if c == best else f1(sc[c], 2)) if c in sc else pend("--") for c in cols]
    d, ci, win = paired(score_pp(E, "policy"), score_pp(E, ref))
    L += [r"score $S$ & " + " & ".join(cells) + f" & {sg(d, 2)}$\\pm${f1(ci, 2)} & {100 * win:.0f}\\%\\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, fname), "w").write("\n".join(L) + "\n")
    out = {"score": sc}
    for c in cols:
        if have(E, c) and c != "policy":
            out[c] = dict(zip(("d", "ci", "win"), paired(score_pp(E, "policy"), score_pp(E, c))))
            out[c]["per"] = {s: dict(zip(("d", "ci", "win"), paired(E["policy"][s][metric], E[c][s][metric]))) for s in VAL}
            out[c]["low"] = {s: float(E[c][s]["low"].mean()) for s in VAL}
    out["policy_low"] = {s: float(E["policy"][s]["low"].mean()) for s in VAL}
    return out


NAMES = {"policy": "Lethe", "UKF+LQR": "UKF+LQR", "UKF+LQR+gamma": r"UKF+LQR+$\gamma$", "UKF-MM": "UKF-MM", "MPC-MM": "MPC-MM",
         "MHE+NMPC": "MHE+NMPC", "oracle-CE": "oracle", "winner": "search winner", "prior": "prior"}

# ---- 30 min, 2000 patients per scenario
E30 = load("eval30")
merge(E30, "mpc30")
nums["main"] = table(E30, ["policy", "UKF+LQR", "UKF+LQR+gamma", "UKF-MM", "MPC-MM", "MHE+NMPC", "oracle-CE"], NAMES, "J", "tab_main.tex")
for c in ("prior", "winner", "PID"):
    nums["main"][c] = dict(zip(("d", "ci", "win"), paired(score_pp(E30, "policy"), score_pp(E30, c))), score=float(score_pp(E30, c).mean()))

# ---- 250 min, 500 patients per scenario, maintenance error
E250 = load("eval250")
merge(E250, "mpc250")
nums["long"] = table(E250, ["policy", "winner", "UKF-MM", "MPC-MM", "MHE+NMPC", "oracle-CE"], NAMES, "Jm", "tab_long.tex", dg=2)

# ---- the controller ladder: what each adds, and the PK-informed observers (covariate check)
# the PK-informed observers (covariate check): a separate run on the same patients, per patient in covariates/<scenario>.npz
Ec = load("covariates")
for s_ in VAL:
    assert np.max(np.abs(Ec["policy"][s_]["J"] - E30["policy"][s_]["J"])) < 1e-9, s_
COV = ("UKF-MM cov0.5", "UKF-MM cov1", "MPC-MM cov1")
for c in COV:
    E30[c] = Ec[c]
CL = [("UKF+LQR", "UKF+LQR", "nominal PK, no delay model"), ("UKF+LQR+gamma", r"UKF+LQR+$\gamma$", r"+ Hill slope $\gamma$"),
      ("UKF-MM", "UKF-MM", "+ delay bank, noise estimate"), ("MPC-MM", "MPC-MM", "+ nonlinear MPC"),
      ("MHE+NMPC", "MHE+NMPC", r"+ MHE of $\gamma$, $k_{e0}$, $E_0$"),
      ("UKF-MM cov0.5", "UKF-MM", "+ half the covariate PK"), ("UKF-MM cov1", "UKF-MM", "+ patient's covariates"),
      ("MPC-MM cov1", "MPC-MM", "+ patient's covariates")]
L = [r"\begin{tabular}{llrrr}", r"\toprule", r"controller & information & $S$ & Lethe$-$it & win\\", r"\midrule"]
for key, nm, info in CL:
    d = dict(zip(("d", "ci", "win"), paired(score_pp(E30, "policy"), score_pp(E30, key))))
    L.append(f"{nm} & {info} & {f1(float(score_pp(E30, key).mean()), 2)} & {sg(d['d'], 2)}$\\pm${f1(d['ci'], 2)} & {100 * d['win']:.0f}\\%\\\\")
L.append(r"\midrule")
L.append(f"Lethe & nominal PK & {f1(nums['main']['score']['policy'], 2)} & & \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_ladder.tex"), "w").write("\n".join(L) + "\n")
nums["cov"] = {c: dict(score=float(score_pp(E30, c).mean()), **dict(zip(("d", "ci", "win"), paired(score_pp(E30, "policy"), score_pp(E30, c)))),
                       per={s_: dict(zip(("d", "ci", "win"), paired(E30["policy"][s_]["J"], E30[c][s_]["J"]))) for s_ in VAL},
                       low={s_: float(E30[c][s_]["low"].mean()) for s_ in VAL}) for c in COV}

# ---- edge boards: median microseconds per step
EDGE = os.path.join(DATA, "edge")
lethe = {h: json.loads([l for l in open(os.path.join(EDGE, f"lethe_{h}.txt")) if l.startswith("{")][0]) for h in ("rpi5", "rpi4", "nano")}
base = {h: {r["baseline"]: r for r in map(json.loads, (l for l in open(os.path.join(EDGE, f"{h}.jsonl")) if l.startswith("{")))}
        for h in ("rpi5", "rpi4", "nano")}
trt = [json.loads(l) for l in open(os.path.join(EDGE, "nano_trt.jsonl"))]
fmt = lambda us: f"{us:.1f}" if us < 100 else (f"{us / 1e3:.1f}\\,ms" if us < 1e5 else f"{us / 1e3:.0f}\\,ms")
L = [r"\begin{tabular}{lrrr}", r"\toprule", r"per 5-s step & Pi 5 & Pi 4 & Jetson Nano\\", r"\midrule"]
L.append("Lethe (C++, float) & " + " & ".join(f"{lethe[h]['float']['ns_per_step_median'] / 1e3:.1f}" for h in base) + r"\\")
L.append("Lethe (C++, double) & " + " & ".join(f"{lethe[h]['double']['ns_per_step_median'] / 1e3:.1f}" for h in base) + r"\\")
for k, nm in (("ukfmm", "UKF-MM"), ("mpcmm", "MPC-MM"), ("mhe", "MHE+NMPC")):
    L.append(f"{nm} & " + " & ".join(fmt(base[h][k]["us_per_step_median"]) + (f" ({fmt(base[h][k]['us_per_step_p99'])})" if k != "ukfmm" else "")
                                     for h in base) + r"\\")
g = min((r for r in trt if r["graph"] and "fp32" in r["engine"]), key=lambda r: r["us_per_step_median"])
L.append(r"Lethe (TensorRT, GPU) & -- & -- & " + f"{g['us_per_step_median']:.0f}" + r"\\")
coral = {r["engine"]: r for r in map(json.loads, open(os.path.join(EDGE, "coral_rpi4.jsonl")))}["edgetpu"]
L.append(r"Lethe (Coral Edge TPU) & -- & " + fmt(coral["us_per_step_median"]) + r" & --\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_edge.tex"), "w").write("\n".join(L) + "\n")
nums["edge"] = dict(lethe={h: {p: lethe[h][p] for p in ("float", "double")} for h in lethe},
                    base={h: {k: {m: base[h][k][m] for m in base[h][k] if m.startswith(("us_", "max_", "ipopt"))} for k in base[h]} for h in base},
                    trt=g, coral=coral)

# ---- long horizon, per 30-min segment (column width)
seg = json.load(open(os.path.join(DATA, "long_segments.json")))
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.4,
                     "lines.linewidth": 1.0, "pdf.fonttype": 42, "font.family": "serif"})
fig, ax = plt.subplots(1, 3, figsize=(3.45, 1.55), sharey=True)
for j, s in enumerate(("nominal", "stimulus", "mixed_hard")):
    for c, col, lab in (("old", "#6f6e69", "with time input"), ("UKF-MM", "#eb6834", "UKF-MM"), ("policy", "#2a78d6", "Lethe"),
                        ("oracle-CE", "#1baf7a", "oracle")):
        if c not in seg[s]:
            continue
        v = seg[s][c]["segs"]
        ax[j].plot(30 * np.arange(len(v)) + 15, v, color=col, marker="o", ms=1.6, lw=0.9, ls="--" if c == "old" else "-",
                   label=lab if j == 0 else None)
    ax[j].axvspan(0, 30, color="#e4e3df", alpha=0.6, lw=0)
    ax[j].set_yscale("log")
    ax[j].set_title({"nominal": "nominal", "stimulus": "signed stimuli", "mixed_hard": "mixed (hard)"}[s], pad=2)
    ax[j].set_xticks([0, 120, 240])
    ax[j].set_xlabel("min", labelpad=1)
ax[0].set_ylabel(r"$J+2V$ per 30 min", labelpad=1)
fig.legend(frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), handlelength=1.5, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.1, 1, 1), pad=0.2, w_pad=0.3)
fig.savefig(os.path.join(OUT, "fig", "long_col.pdf"))
nums["segments"] = {s: {c: seg[s][c] for c in seg[s]} for s in seg}
json.dump(nums, open(os.path.join(OUT, "numbers_ecc.json"), "w"), indent=1)
print("ok; mpc250:", have(E250, "MPC-MM"))
