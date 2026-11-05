"""Every table and figure of the Lethe paper, from the stored outputs of the run.

  python analyze.py DATA_DIR LATEX_DIR

DATA_DIR holds (copied from the run folder, nothing recomputed except the trace quantiles and the long-horizon segments):
  search/        search.py output: log.jsonl (every trial's snapshots and rung ends), manifest.json, results.json, prior_pool.json
  finetune/      finetune.py output: log.jsonl (validation of the 9 starts), results.json, best_model.pkl
  eval30/        evaluate.py, 30 min, 2000 patients per scenario (validation, stress, stimulus times): <scenario>.npz with
                 <controller>__<metric> per patient, results.json; traces_q.npz: BIS / infusion quantiles of the same closed loops (traces.py)
  eval250/       the same, 250 min, 500 patients per validation scenario
  mpc30/, mpc250/  evaluate.py with the MPC baselines on the same patients (Lethe and UKF-MM again: the check that they are the same)
  long_segments.json  cost per 30-min segment over 250 min, the earlier model with the time input and Lethe on the same patients (traces.py)
  model_info.json     prior design, physics modes, S4D memories and parameter counts of the model (model_info.py)
  search.log, finetune.log  stdout of the stages
Writes LATEX_DIR/tab_*.tex, LATEX_DIR/fig/*.pdf and LATEX_DIR/numbers.json.
"""
import os, re, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA, OUT = sys.argv[1], sys.argv[2]
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)
DT = 5.0 / 60.0

VAL = ["nominal", "stimulus", "fast_drift", "delay_30s", "delay_60s", "gamma_wide", "pk_shift", "target_range", "target_steps", "noise_steps",
       "mixed_hard"]
W = np.array([2.0] + [1.0] * 9 + [2.0])
STRESS = ["gamma_ood_lo", "gamma_ood_hi", "pk_shift_ood", "target_42", "target_65", "noise_8", "target_steps_ood", "noise_bursts", "everything_ood",
          "stimulus_neg", "stimulus_signed"]
STIM = ["stim_at_10", "stim_at_15", "stim_at_20", "stim_at_25"]
NICE = {"nominal": "nominal", "stimulus": "stimulus", "fast_drift": "fast drift", "delay_30s": "delay 30 s", "delay_60s": "delay 60 s",
        "gamma_wide": r"wide $\gamma$", "pk_shift": "PK shift", "target_range": "target range", "target_steps": "target steps",
        "noise_steps": "noise steps", "mixed_hard": "mixed (hard)", "gamma_ood_lo": r"$\gamma=1$", "gamma_ood_hi": r"$\gamma=6$",
        "pk_shift_ood": "PK shift 0.9", "target_42": "target 42", "target_65": "target 65", "noise_8": r"noise $\sigma_y=8$",
        "target_steps_ood": "target steps 42--65", "noise_bursts": "noise bursts", "everything_ood": "everything OOD",
        "stimulus_neg": "stimulus lowering $C_{50}$", "stimulus_signed": "1--3 signed stimuli"}
PLAIN = {k: v.replace("$", "")
         .replace("\\gamma", "γ").replace("\\sigma_y", "σy").replace("--", "–").replace("C_{50}", "C50") for k, v in NICE.items()}
CNAME = {"policy": "Lethe", "winner": "search winner", "prior": "prior (untrained)", "UKF-MM": "UKF-MM",
         "MPC-MM": "MPC-MM", "MHE+NMPC": "MHE+NMPC", "UKF+LQR+gamma": r"UKF+LQR+$\gamma$", "UKF+LQR": "UKF+LQR", "PID": "PID",
         "oracle-CE": "oracle-CE"}
MPC = ["MPC-MM", "MHE+NMPC"]
nums = {}

# palette (categorical slots in fixed order), neutral gray for references
C1, C2, C3, C4, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a35bd1", "#6f6e69"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "axes.edgecolor": "#8a8984", "lines.linewidth": 1.4, "pdf.fonttype": 42, "font.family": "serif"})


def paired(a, b):
    d = a - b
    return float(d.mean()), float(1.96 * d.std(ddof=1) / math.sqrt(len(d))), float((d < 0).mean())


def f2(x, n=1):
    return f"{x:.{n}f}"


def sgn(x, n=1):
    return f"{x:+.{n}f}".replace("-", "$-$")


def save(name, lines):
    open(os.path.join(OUT, name), "w").write("\n".join(lines) + "\n")


def load_eval(folder, scens):
    """per-patient metrics: E[controller][scenario][metric]"""
    E = {}
    for s in scens:
        p = os.path.join(DATA, folder, f"{s}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        for k in z.files:
            c, m = k.split("__")
            E.setdefault(c, {}).setdefault(s, {})[m] = z[k]
    return E


def merge_mpc(E, folder, scens, tag):
    """add the MPC baselines (evaluated in a separate run on the same patients) after checking that Lethe and UKF-MM agree per patient"""
    Em = load_eval(folder, scens)
    chk = {}
    for s in scens:
        if s not in Em.get("policy", {}):
            continue
        chk[s] = {c: float(np.max(np.abs(Em[c][s]["J"] - E[c][s]["J"]))) for c in ("policy", "UKF-MM")}
        for c in MPC:
            if s in Em.get(c, {}):
                E.setdefault(c, {})[s] = Em[c][s]
    nums[f"mpc_check_{tag}"] = chk
    return all(s in E.get(c, {}) for c in MPC for s in scens)


sc_part = lambda E, c, s: E[c][s]["J"] + 2.0 * E[c][s]["viol"]
score_pp = lambda E, c: sum(w * sc_part(E, c, s) for w, s in zip(W, VAL)) / W.sum()      # per-patient score; scenarios are independent draws


def summarize(E, ctrls, scens):
    ev = {}
    for c in ctrls:
        if c not in E:
            continue
        ev[c] = dict(per={s: {m: float(np.nanmean(v)) for m, v in E[c][s].items()} for s in scens if s in E[c]})
        if all(s in E[c] for s in VAL):
            ev[c]["score"] = float(score_pp(E, c).mean())
    for c in ev:
        if c == "policy":
            continue
        if "score" in ev[c]:
            ev[c]["score_diff"], ev[c]["score_diff_ci"], ev[c]["score_win"] = paired(score_pp(E, "policy"), score_pp(E, c))
        for s in ev[c]["per"]:
            ev[c]["per"][s]["diff"], ev[c]["per"][s]["ci"], ev[c]["per"][s]["win"] = paired(E["policy"][s]["J"], E[c][s]["J"])
            ev[c]["per"][s]["dm"], ev[c]["per"][s]["dm_ci"], ev[c]["per"][s]["win_m"] = paired(E["policy"][s]["Jm"], E[c][s]["Jm"])
    return ev


CTRL = ["policy", "winner", "prior", "UKF-MM", "MPC-MM", "MHE+NMPC", "UKF+LQR+gamma", "UKF+LQR", "PID", "oracle-CE"]
REF = ("oracle-CE",)                     # not in the "best causal controller" bold


def eval_table(ev, scens, fname, cols, with_score, metric="J"):
    """mean `metric` per controller and scenario, paired difference vs UKF-MM (95% CI), win share, time below 40"""
    cols = [c for c in cols if c in ev and all(s in ev[c]["per"] for s in scens)]
    causal = [c for c in cols if c not in REF]
    n, dg = len(cols), (1 if metric == "J" else 2)
    L = [r"\begin{tabular}{l" + "r" * n + "rrrr}", r"\toprule",
         "scenario & " + " & ".join(CNAME[c] for c in cols) + r" & $\Delta$ vs UKF-MM & win & \multicolumn{2}{c}{\% time $<40$}\\",
         rf"\cmidrule(lr){{{n + 4}-{n + 5}}}", " & " * (n + 3) + r"Lethe & UKF-MM\\", r"\midrule"]
    for s in scens:
        Js = {c: ev[c]["per"][s][metric] for c in cols}
        best = min(causal, key=lambda c: Js[c])
        cells = [(r"\textbf{" + f2(Js[c], dg) + "}") if c == best else f2(Js[c], dg) for c in cols]
        p = ev["UKF-MM"]["per"][s]
        d, ci, win = (p["diff"], p["ci"], p["win"]) if metric == "J" else (p["dm"], p["dm_ci"], p["win_m"])
        L.append(f"{NICE[s]} & " + " & ".join(cells) + f" & {sgn(d, dg)}$\\pm${f2(ci, dg)} & {100 * win:.0f}\\% & "
                 f"{f2(ev['policy']['per'][s]['low'], 2)} & {f2(ev['UKF-MM']['per'][s]['low'], 2)}\\\\")
    if with_score:
        L.append(r"\midrule")
        best = min(causal, key=lambda c: ev[c]["score"])
        cells = [(r"\textbf{" + f2(ev[c]["score"]) + "}") if c == best else f2(ev[c]["score"]) for c in cols]
        L.append(r"score $S$ & " + " & ".join(cells) + f" & {sgn(ev['UKF-MM']['score_diff'])}$\\pm${f2(ev['UKF-MM']['score_diff_ci'])} & "
                 f"{100 * ev['UKF-MM']['score_win']:.0f}\\% & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    save(fname, L)


# ======================================================== evaluation, 30 min (2000 patients per scenario, the same patients for every controller)
E30 = load_eval("eval30", VAL + STRESS + STIM)
have_mpc30 = merge_mpc(E30, "mpc30", VAL, "30")
ev = summarize(E30, CTRL, VAL + STRESS + STIM)
nums["eval"] = ev
nums["win_vs_ukfmm_val_J"] = float(np.mean(np.concatenate([E30["policy"][s]["J"] < E30["UKF-MM"][s]["J"] for s in VAL])))

MAIN = ["policy", "winner", "prior", "UKF-MM", "MPC-MM", "MHE+NMPC", "UKF+LQR+gamma", "UKF+LQR", "PID", "oracle-CE"]
eval_table(ev, VAL, "tab_val.tex", MAIN, True)
eval_table(ev, STRESS, "tab_stress.tex", [c for c in MAIN if c not in MPC], False)

# ---- paired differences of the score against every controller
L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Lethe $-$ & $\Delta S$ & 95\% CI & Lethe better\\", r"\midrule"]
for c in CTRL[1:]:
    if c in ev and "score" in ev[c]:
        L.append(f"{CNAME[c]} & {sgn(ev[c]['score_diff'], 2)} & $\\pm${f2(ev[c]['score_diff_ci'], 2)} & {100 * ev[c]['score_win']:.0f}\\%\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
save("tab_score.tex", L)

# ---- phase decomposition, averaged over the validation scenarios
ph = {}
IND_N, N30 = int(np.sum(np.arange(360) * DT <= 10.0)), 360
for c in ["policy", "winner", "UKF-MM", "MPC-MM", "MHE+NMPC", "UKF+LQR+gamma", "UKF+LQR", "oracle-CE"]:
    if c not in E30 or not all(s in E30[c] for s in VAL):
        continue
    g = lambda m: float(np.mean([np.nanmean(E30[c][s][m]) for s in VAL]))
    ph[c] = dict(J=g("J"), J_ind=g("J_ind"), Jm=g("Jm"), t_ind=g("t_ind"), band=g("band"), drug=g("drug"), low=g("low"))
    ph[c]["ind_share"] = (ph[c]["J_ind"] * IND_N / N30) / ph[c]["J"]
nums["phase"] = ph
L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
     r"controller & $J$ & $J_{\mathrm{ind}}$ & $J_m$ & induction share & $t_{\mathrm{ind}}$ (min) & in band (\%) & drug (mg)\\", r"\midrule"]
for c, d in ph.items():
    L.append(f"{CNAME[c]} & {f2(d['J'])} & {f2(d['J_ind'])} & {f2(d['Jm'], 2)} & {100 * d['ind_share']:.0f}\\% & {f2(d['t_ind'], 2)} & "
             f"{f2(d['band'], 1)} & {f2(d['drug'], 0)}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
save("tab_phase.tex", L)

# ---- stimulus timing (a check: nothing in training or selection fixes the time)
TC = ["policy", "winner", "UKF-MM", "UKF+LQR", "oracle-CE"]
L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"stimulus at & 10 min & 15 min & 20 min & 25 min & range\\", r"\midrule"]
for c in TC:
    v = [ev[c]["per"][s]["J"] for s in STIM]
    L.append(f"{CNAME[c]} & " + " & ".join(f2(x) for x in v) + f" & {f2(np.ptp(v), 2)}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
save("tab_timing.tex", L)
nums["timing_spread"] = {c: float(np.ptp([ev[c]["per"][s]["J"] for s in STIM])) for c in TC}

# ---- tails
tails = {}
for s in ["mixed_hard", "everything_ood", "delay_60s", "delay_30s"]:
    tails[s] = {c: dict(p50=float(np.percentile(E30[c][s]["J"], 50)), p95=float(np.percentile(E30[c][s]["J"], 95)),
                       low=float(np.mean(E30[c][s]["low"])))
                for c in ["policy", "winner", "UKF-MM", "MPC-MM", "MHE+NMPC", "UKF+LQR+gamma"] if c in E30 and s in E30[c]}
nums["tails"] = tails

# ================================================================ evaluation, 250 min (500 patients per validation scenario)
E250 = load_eval("eval250", VAL)
have_mpc250 = merge_mpc(E250, "mpc250", VAL, "250")
ev250 = summarize(E250, CTRL, VAL)
nums["eval250"] = ev250
LC = [c for c in ["policy", "winner", "UKF-MM", "MPC-MM", "MHE+NMPC", "UKF+LQR+gamma", "UKF+LQR", "oracle-CE"] if c in ev250]
eval_table(ev250, VAL, "tab_long.tex", LC, True, metric="Jm")
nums["long_band"] = {c: float(np.mean([ev250[c]["per"][s]["band"] for s in VAL])) for c in LC if all(s in ev250[c]["per"] for s in VAL)}

# ---- the earlier model with the time input, 250 min, per 30-min segment (the same patients for every controller)
seg = json.load(open(os.path.join(DATA, "long_segments.json")))
nums["long_segments"] = seg
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
for j, s in enumerate(("nominal", "stimulus", "mixed_hard")):
    for c, col, lab in (("old", GRAY, "with time input (earlier)"), ("UKF-MM", C2, "UKF-MM"), ("policy", C1, "Lethe"),
                        ("oracle-CE", C3, "oracle-CE")):
        if c not in seg[s]:
            continue
        v = seg[s][c]["segs"]
        ax[j].plot(30 * np.arange(len(v)) + 15, v, color=col, marker="o", ms=2.5, lw=1.0, ls="--" if c == "old" else "-",
                   label=lab if j == 0 else None)
    ax[j].axvspan(0, 30, color="#e4e3df", alpha=0.6, lw=0)
    ax[j].set_yscale("log")
    ax[j].set_title({"nominal": "nominal", "stimulus": "1–3 signed stimuli", "mixed_hard": "mixed (hard)"}[s])
    ax[j].set_xlabel("time (min)")
    ax[j].set_xticks([0, 60, 120, 180, 240])
ax[0].set_ylabel(r"$J+2V$ per 30-min segment")
fig.legend(frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(FIG, "long.pdf"))
plt.close(fig)

# ================================================================ search
LOG = [json.loads(l) for l in open(os.path.join(DATA, "search", "log.jsonl"))]
SR = json.load(open(os.path.join(DATA, "search", "results.json")))
MAN = json.load(open(os.path.join(DATA, "search", "manifest.json")))
cfgs0 = {t["id"]: t["cfg"] for t in MAN["trials"]}
stages, finals = ["rung0", "rung1", "rung2"], ["final-s0", "final-s1", "final-s2"]
ends = {st: {r["trial"]: r for r in LOG if r["stage"] == st and "score" in r} for st in stages + finals}
FIN = [c["parent"] for c in sorted(SR["configs"], key=lambda c: c["mean"])]                 # finalists, best re-rank first
srch = {}
for st, its in zip(stages, (300, 900, 2700)):
    s = np.array([r["score"] for r in ends[st].values()])
    b = np.array([r["best_score"] for r in ends[st].values()])
    srch[st] = dict(n=len(s), it=its, best=float(s.min()), median=float(np.median(s)), best_snap=float(b.min()), median_snap=float(np.median(b)),
                    snap_not_end=float(np.mean([r["best_it"] != r["it"] for r in ends[st].values()])),
                    snap_gain_median=float(np.median(s - b)))
seed_id = lambda tr, k: tr + 100000 * k
srch["final"] = {f"{tr}-s{k}": dict(end=ends[finals[k]][seed_id(tr, k)]["score"], best=ends[finals[k]][seed_id(tr, k)]["best_score"],
                                    best_it=ends[finals[k]][seed_id(tr, k)]["best_it"]) for tr in FIN for k in range(3)}
srch["rerank"] = SR["rerank"]
srch["configs"] = SR["configs"]
srch["winner"] = dict(model=SR["best_model"], parent=SR["best_parent"], snapshot_it=SR["snapshot_it"])
srch["test"] = dict(score=SR["test_score"], prior=SR["base_prior_score"], improvement=SR["test_improvement"], ci=SR["test_improvement_ci"])
slog = open(os.path.join(DATA, "search.log")).read()
mins = {}
for st in stages + finals:
    m = re.findall(rf"\[{st}\] (\d+)/\1 trials, ([0-9.]+) min", slog)
    if m:
        mins[st] = float(m[-1][1])
ps = re.search(r"prior screen: (\d+) designs in (\d+)s, init score best ([0-9.]+) median ([0-9.]+); base design rank (\d+)", slog)
srch["minutes"] = dict(prior_screen=int(ps.group(2)) / 60, **{st: mins[st] for st in stages}, final=sum(mins[f] for f in finals))
srch["prior_screen"] = dict(n=int(ps.group(1)), best=float(ps.group(3)), median=float(ps.group(4)), base_rank=int(ps.group(5)))
srch["trial0"] = {st: ends[st][0]["score"] for st in stages if 0 in ends[st]}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


for a_, b_ in (("rung0", "rung1"), ("rung1", "rung2")):
    ids = sorted(set(ends[a_]) & set(ends[b_]))
    srch[f"spearman_{a_}_{b_}"] = spearman([ends[a_][i]["score"] for i in ids], [ends[b_][i]["score"] for i in ids])
rank = {st: {r["trial"]: i + 1 for i, r in enumerate(sorted(ends[st].values(), key=lambda r: r["score"]))} for st in stages}
srch["finalist_rank"] = {tr: {st: rank[st][tr] for st in stages} for tr in FIN}
pool = json.load(open(os.path.join(DATA, "search", "prior_pool.json")))
srch["prior_pool"] = dict(n=len(pool["pool"]), best=pool["scores"][0], worst=pool["scores"][-1])
# rung-0 marginals (median score per value) as search.py printed them: the manifest keeps only the final models' configurations
marg = {}
for line in slog.split("rung-0 marginals")[1].splitlines()[1:]:
    mm = re.match(r"\s+(\w+)\s+(.*)", line)
    if not mm or ":" not in mm.group(2):
        break
    marg[mm.group(1)] = {k: float(v) for k, v in re.findall(r"(\S+): ([0-9.]+)", mm.group(2))}
srch["marginals_rung0"] = marg
top10 = sorted(ends["rung2"].values(), key=lambda r: r["score"])[:10]
srch["rung2_top10"] = dict(lo=top10[0]["score"], hi=top10[-1]["score"])
nums["search"] = srch

L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"stage & trials & iterations & best & median & wall time (min)\\", r"\midrule",
     f"prior screen & {srch['prior_screen']['n'] - 1} designs & 0 & {srch['prior_screen']['best']:.1f} & {srch['prior_screen']['median']:.1f} & "
     f"{srch['minutes']['prior_screen']:.1f}\\\\"]
for st, nm in zip(stages, ("rung 0", "rung 1", "rung 2")):
    d = srch[st]
    L.append(f"{nm} & {d['n']} & {d['it']} & {d['best']:.2f} & {d['median']:.2f} & {srch['minutes'][st]:.0f}\\\\")
fin = np.array([v["best"] for v in srch["final"].values()])
L.append(f"final (3 configs $\\times$ 3 seeds) & 9 & 5700 & {fin.min():.2f} & {np.median(fin):.2f} & {srch['minutes']['final']:.0f}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
save("tab_search.tex", L)

# ================================================================ fine-tuning (every final-stage model)
FL = [json.loads(l) for l in open(os.path.join(DATA, "finetune", "log.jsonl"))]
FR = json.load(open(os.path.join(DATA, "finetune", "results.json")))
starts = [r for r in FR["rerank"] if r["best_val"] is not None]
# the log holds the populations one after the other (30 rows each); a member is matched to its start by its best validation and iteration
curves = {}
for pop in (FL[i:i + 30] for i in range(0, len(FL), 30)):
    sc = np.array([r["scores"] for r in pop]); its = np.array([r["it"] for r in pop])
    for j in range(sc.shape[1]):
        k = int(np.argmin(sc[:, j]))
        cand = [r for r in starts if r["best_it"] == its[k] and r["name"] not in curves]
        nm = min(cand, key=lambda r: abs(r["best_val"] - sc[k, j]))["name"]
        curves[nm] = (its, sc[:, j])
parse = lambda nm: (int(nm.split()[1]), int(nm.split()[3].rstrip(",")))
rr_ft = {}
for r in starts:
    rr_ft.setdefault(parse(r["name"])[0], []).append(r["mean"])
flog = open(os.path.join(DATA, "finetune.log")).read()
m = re.search(r"trained in ([0-9.]+) min", flog)
nums["finetune"] = dict(model=FR["model"], iters=FR["iters"], test=FR["test_score"], winner=FR["winner_score"], prior=FR["base_prior_score"],
                        gain=FR["gain_over_winner"], gain_ci=FR["gain_over_winner_ci"], impr=FR["test_improvement"],
                        impr_ci=FR["test_improvement_ci"],
                        rerank=FR["rerank"], rerank_by_config={str(k): float(np.mean(v)) for k, v in rr_ft.items()},
                        minutes=float(m.group(1)) if m else None)
L = [r"\begin{tabular}{lrrr}", r"\toprule", r"start & best validation & at iteration & re-rank\\", r"\midrule"]
for r in sorted(FR["rerank"], key=lambda r: r["mean"]):
    if r["best_val"] is None:
        L.append(f"search winner (no fine-tuning) & -- & -- & {r['mean']:.3f}\\\\")
    else:
        c_, s_ = parse(r["name"])
        L.append(f"config {c_}, seed {s_} & {r['best_val']:.2f} & {r['best_it']} & {r['mean']:.3f}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
save("tab_finetune.tex", L)

# finalists: configs, rung ranks, final-stage best snapshots, search re-rank, fine-tuned re-rank
cfgs = {i: cfgs0[i] for i in FIN}
rr = {}
for r in SR["rerank"]:
    rr.setdefault(r["parent"], []).append(r["mean"])
L = [r"\begin{tabular}{l" + "r" * len(FIN) + "}", r"\toprule", "& " + " & ".join(f"config {i}" for i in FIN) + r"\\", r"\midrule"]
rows = [("optimizer, schedule", lambda c: f"{c['opt']}, {c['sched']}"), ("base learning rate", lambda c: "$%.2f\\times10^{-3}$" % (1e3 * c["lr"])),
        ("$H$ / $N_s$ / $L$ / head", lambda c: f"{c['H']} / {c['NS']} / {c['L']} / {c['head']}"),
        ("extra physics modes", lambda c: f"{c['n_extra']}"),
        ("$w_{\\mathrm{low}}$, $\\rho_u$ (search)", lambda c: f"{c['w_low']:.2f}, {c['rho_u']:.2f}")]
for name, f in rows:
    L.append(name + " & " + " & ".join(f(cfgs[i]) for i in FIN) + r"\\")
for st in stages:
    L.append(f"rank at rung {st[-1]} (of {srch[st]['n']}) & " + " & ".join(str(srch["finalist_rank"][i][st]) for i in FIN) + r"\\")
L.append("final stage, best snapshot, 3 seeds & "
         + " & ".join("/".join(f"{srch['final'][f'{i}-s{k}']['best']:.2f}" for k in range(3)) for i in FIN) + r"\\")
L.append("search re-rank, mean of 3 seeds & " + " & ".join(f"{np.mean(rr[i]):.3f}" for i in FIN) + r"\\")
L.append("fine-tuned re-rank (120 min), mean & " + " & ".join(f"{np.mean(rr_ft[i]):.3f}" for i in FIN) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
save("tab_finalists.tex", L)
nums["rerank_by_config"] = {str(k): dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in rr.items()}
nums["finalists"] = FIN
nums["finalist_cfgs"] = {str(k): v for k, v in cfgs.items()}

# ================================================================ the model
nums["model"] = json.load(open(os.path.join(DATA, "model_info.json")))

# ================================================================ figures
# --- search: every trial's end-of-rung score per stage, survivors highlighted, the finalists' path
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
rng = np.random.default_rng(0)
FC = dict(zip(FIN, (C2, C3, C4)))
for j, st in enumerate(stages):
    s = np.array([r["score"] for r in ends[st].values()])
    nxt = set(ends[stages[j + 1]]) if j + 1 < len(stages) else set(FIN)
    surv = np.array([tr in nxt for tr in ends[st]])
    x = j + rng.uniform(-0.28, 0.28, len(s))
    ax[0].scatter(x[~surv], s[~surv], s=2, color=GRAY, alpha=0.35, lw=0)
    ax[0].scatter(x[surv], s[surv], s=3, color=C1, alpha=0.8, lw=0)
for tr in FIN:
    v = [srch["final"][f"{tr}-s{k}"]["best"] for k in range(3)]
    ax[0].plot(range(4), [ends[st][tr]["score"] for st in stages] + [np.mean(v)], color=FC[tr], lw=1.0, marker="o", ms=2.5, label=f"config {tr}")
    ax[0].scatter([3] * 3, v, s=6, color=FC[tr], zorder=3)
ax[0].set_xticks(range(4), [f"rung 0\n300 it, {srch['rung0']['n']}", f"rung 1\n900 it, {srch['rung1']['n']}",
                            f"rung 2\n2700 it, {srch['rung2']['n']}", "final\n5700 it, 3×3"])
ax[0].set_yscale("log")
ax[0].set_ylim(76, 135)
ax[0].set_yticks([78, 80, 85, 90, 100, 120], ["78", "80", "85", "90", "100", "120"])
ax[0].minorticks_off()
ax[0].set_ylabel("validation score $S$")
ax[0].legend(loc="upper right", frameon=False)
ax[0].set_title("successive halving (blue: promoted)")
ids = sorted(set(ends["rung0"]) & set(ends["rung1"]))
ax[1].scatter([ends["rung0"][i]["score"] for i in ids], [ends["rung1"][i]["score"] for i in ids], s=4, color=C1, alpha=0.6, lw=0)
for tr in FIN:
    ax[1].scatter(ends["rung0"][tr]["score"], ends["rung1"][tr]["score"], s=18, color=FC[tr], zorder=3)
ax[1].set_xlabel("score at rung 0 (300 it)")
ax[1].set_ylabel("score at rung 1 (900 it)")
ax[1].set_title(f"rank agreement, Spearman {srch['spearman_rung0_rung1']:.2f}")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "search.pdf"))
plt.close(fig)

# --- fine-tuning: validation of the 9 starts
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for nm, (its, sc) in sorted(curves.items()):
    c_, s_ = parse(nm)
    ax.plot(its, sc, color=FC[c_], lw=0.9, ls=("-", "--", ":")[s_], label=f"config {c_}" if s_ == 0 else None)
ax.axvline(1000, color=GRAY, lw=0.6, ls=":")
ax.set_xlabel("fine-tuning iteration")
ax.set_ylabel("validation score $S$ (120 min)")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=0.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "finetune.pdf"))
plt.close(fig)

# --- paired difference against UKF-MM per scenario
fig, ax = plt.subplots(figsize=(3.4, 4.0))
scs = VAL + STRESS
y = np.arange(len(scs))[::-1]
d = np.array([ev["UKF-MM"]["per"][s]["diff"] for s in scs]); ci = np.array([ev["UKF-MM"]["per"][s]["ci"] for s in scs])
ax.errorbar(d, y, xerr=ci, fmt="o", ms=3, color=C1, ecolor=C1, elinewidth=1, capsize=1.5)
ax.axvline(0, color=GRAY, lw=0.8)
ax.axhline(len(STRESS) - 0.5, color=GRAY, lw=0.6, ls="--")
ax.set_yticks(y, [PLAIN[s] for s in scs])
ax.set_xscale("symlog", linthresh=5)
ax.set_xlim(min(-45, 1.2 * float(np.min(d - ci))), max(8, 1.2 * float(np.max(d + ci))))
ax.set_xticks([-30, -10, -5, 0, 5], ["−30", "−10", "−5", "0", "5"])
ax.set_xlabel("Lethe − UKF-MM, mean J (95% CI)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "paired.pdf"))
plt.close(fig)

# --- traces: median and IQR of BIS and infusion, 30 min
t = np.arange(360) * DT
Q = np.load(os.path.join(DATA, "eval30", "traces_q.npz"))
show = ["nominal", "delay_60s", "mixed_hard"]
fig, ax = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True)
for j, s in enumerate(show):
    for c, col in (("UKF+LQR", C3), ("UKF-MM", C2), ("policy", C1)):
        for i, nm in enumerate(("b", "u")):
            q = Q[f"{s}__{c}__{nm}"]
            ax[i, j].plot(t, q[2], color=col, lw=1.0, label=CNAME[c] if (i == 0 and j == 0) else None)
            ax[i, j].fill_between(t, q[1], q[3], color=col, alpha=0.15, lw=0)
    ax[0, j].axhline(40, color=GRAY, lw=0.6, ls=":")
    ax[0, j].set_title(PLAIN[s])
    ax[0, j].set_ylim(30, 100)
    ax[1, j].set_yscale("symlog", linthresh=10)
    ax[1, j].set_ylim(0, 105)
    ax[1, j].set_yticks([0, 5, 10, 20, 50, 100], ["0", "5", "10", "20", "50", "100"])
    ax[1, j].set_xlabel("time (min)")
    ax[1, j].set_xscale("symlog", linthresh=2)
    ax[1, j].set_xticks([0, 1, 2, 5, 10, 30], ["0", "1", "2", "5", "10", "30"])
ax[0, 0].set_ylabel("true BIS")
ax[1, 0].set_ylabel("infusion (mg/min)")
ax[0, 0].legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "traces.pdf"))
plt.close(fig)
# the median patient's induction (numbers for the text)
win_ = lambda a, b_: (t >= a) & (t < b_)
prof = {}
for s in show:
    prof[s] = {}
    for c in ("policy", "UKF-MM", "UKF+LQR"):
        u, b = Q[f"{s}__{c}__u"][2], Q[f"{s}__{c}__b"][2]
        off = np.flatnonzero(u < 99.0)
        prof[s][c] = dict(full_until=float(t[off[0]]) if off.size else None, min_bis=float(b.min()), maint_rate=float(np.median(u[win_(10, 30)])),
                          low_until=float(t[np.flatnonzero(u[:60] < 1.0)[-1]]) if np.any(u[:60] < 1.0) else None)
nums["induction_profile"] = prof

# --- long horizon: median and IQR of BIS and infusion, 250 min
Q2 = np.load(os.path.join(DATA, "eval250", "traces_q.npz"))
t2 = np.arange(3000) * DT
fig, ax = plt.subplots(2, 3, figsize=(7.0, 3.4), sharex=True)
for j, s in enumerate(("nominal", "stimulus", "mixed_hard")):
    for c, col in (("oracle-CE", C3), ("UKF-MM", C2), ("policy", C1)):
        for i, nm in enumerate(("b", "u")):
            q = Q2[f"{s}__{c}__{nm}"]
            ax[i, j].plot(t2, q[2], color=col, lw=0.8, label=CNAME[c] if (i == 0 and j == 0) else None)
            ax[i, j].fill_between(t2, q[1], q[3], color=col, alpha=0.15, lw=0)
    for i in range(2):
        ax[i, j].axvline(30, color=GRAY, lw=0.6, ls="--")
    ax[0, j].axhline(40, color=GRAY, lw=0.6, ls=":")
    ax[0, j].set_title(PLAIN[s])
    ax[0, j].set_ylim(35, 65)
    ax[1, j].set_ylim(0, 40)
    ax[1, j].set_xlabel("time (min)")
    ax[1, j].set_xlim(3, 250)
ax[0, 0].set_ylabel("true BIS")
ax[1, 0].set_ylabel("infusion (mg/min)")
leg = ax[0, 0].legend(frameon=False, loc="lower right")
for h in leg.legend_handles:
    h.set_linewidth(2.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "long_traces.pdf"))
plt.close(fig)

nums["have_mpc"] = dict(eval30=have_mpc30, eval250=have_mpc250)
os.makedirs(os.path.join(DATA, "..", "analysis"), exist_ok=True)
# ================================================================ covariates: UKF-MM / MPC-MM initialized with the patient's individual PK
if os.path.isdir(os.path.join(DATA, "covariates")):
    Ec = load_eval("covariates", VAL)
    for s in VAL:
        assert np.max(np.abs(Ec["policy"][s]["J"] - E30["policy"][s]["J"])) < 1e-9, s
    CV = [("UKF-MM", "UKF-MM"), ("UKF-MM cov0.5", r"UKF-MM, $\frac12$cov"), ("UKF-MM cov1", "UKF-MM, cov"), ("MPC-MM cov1", "MPC-MM, cov")]
    L = [r"\begin{tabular}{l" + "r" * (1 + len(CV)) + "rrr}", r"\toprule",
         "scenario & Lethe & " + " & ".join(n for _, n in CV) + r" & \multicolumn{3}{c}{Lethe $-$ (95\% CI)}\\",
         r"\cmidrule(lr){" + f"{3 + len(CV)}-{5 + len(CV)}" + "}", " & " * (1 + len(CV)) + r" & UKF-MM, cov & MPC-MM, cov & win\\", r"\midrule"]
    for s in VAL:
        js = [Ec[c][s]["J"].mean() for c, _ in CV]
        d1, c1, _ = paired(Ec["policy"][s]["J"], Ec["UKF-MM cov1"][s]["J"])
        d2, c2, w2 = paired(Ec["policy"][s]["J"], Ec["MPC-MM cov1"][s]["J"])
        L.append(f"{NICE[s]} & {f2(Ec['policy'][s]['J'].mean())} & " + " & ".join(f2(v) for v in js)
                 + f" & {sgn(d1)}$\\pm${f2(c1)} & {sgn(d2)}$\\pm${f2(c2)} & {100 * w2:.0f}\\%\\\\")
    L.append(r"\midrule")
    sc = lambda c: sum(w * (Ec[c][s]["J"] + 2 * Ec[c][s]["viol"]) for w, s in zip(W, VAL)) / W.sum()
    d1, c1, _ = paired(sc("policy"), sc("UKF-MM cov1"))
    d2, c2, w2 = paired(sc("policy"), sc("MPC-MM cov1"))
    L.append(r"score $S$ & " + f"{f2(sc('policy').mean(), 2)} & " + " & ".join(f2(sc(c).mean(), 2) for c, _ in CV)
             + f" & {sgn(d1, 2)}$\\pm${f2(c1, 2)} & {sgn(d2, 2)}$\\pm${f2(c2, 2)} & {100 * w2:.0f}\\%\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    save("tab_cov.tex", L)
    nums["cov"] = {c: float(sc(c).mean()) for c in ["policy"] + [c for c, _ in CV]}

# ================================================================ edge boards (edge.py results: median / p99 per step, max error vs the reference)
EDGE = os.path.join(DATA, "edge")
if os.path.isdir(EDGE):
    jl = lambda f: [json.loads(l) for l in open(os.path.join(EDGE, f)) if l.startswith("{")]
    boards = ("rpi5", "rpi4", "nano")
    leth = {h: jl(f"lethe_{h}.txt")[0] for h in boards}
    base = {h: {r["baseline"]: r for r in jl(f"{h}.jsonl")} for h in boards}
    us = lambda v: f"{v:.1f}" if v < 1e3 else f"{v / 1e3:.1f}\\,ms"
    L = [r"\begin{tabular}{l" + "rr" * len(boards) + "r}", r"\toprule",
         "per step & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for n in ("Raspberry Pi 5", "Raspberry Pi 4", "Jetson Nano")) + r" & error\\",
         " & " + " & ".join(["median & p99"] * len(boards)) + r" & \\", r"\midrule"]
    for p, nm in (("float", "Lethe, C++ float"), ("double", "Lethe, C++ double")):
        L.append(f"{nm} & " + " & ".join(f"{leth[h][p]['ns_per_step_median'] / 1e3:.1f} & {leth[h][p]['ns_per_step_p99'] / 1e3:.1f}" for h in boards)
                 + f" & {max(leth[h][p]['max_abs_err_vs_jax'] for h in boards):.1e}\\\\")
    for k, nm in (("ukfmm", "UKF-MM"), ("mpcmm", "MPC-MM"), ("mhe", "MHE+NMPC")):
        L.append(f"{nm} & " + " & ".join(f"{us(base[h][k]['us_per_step_median'])} & {us(base[h][k]['us_per_step_p99'])}" for h in boards)
                 + f" & {max(base[h][k]['max_abs_err_vs_python'] for h in boards):.1e}\\\\")
    trt = [r for r in jl("nano_trt.jsonl") if r["graph"]]
    for r in trt:
        if r["engine"] in ("policy_fp32.plan", "policy_fp16.plan"):
            L.append(f"Lethe, TensorRT {r['engine'][7:11]} (GPU) & -- & -- & -- & -- & {r['us_per_step_median']:.0f} & {r['us_per_step_p99']:.0f}"
                     f" & {r['max_abs_err_vs_jax']:.1e}\\\\")
    co = {r["engine"]: r for r in jl("coral_rpi4.jsonl")}
    for e, nm in (("cpu_int8", "Lethe, int8 split, CPU (Python)"), ("edgetpu", "Lethe, int8 split, Coral TPU (Python)")):
        L.append(f"{nm} & -- & -- & {us(co[e]['us_per_step_median'])} & {us(co[e]['us_per_step_p99'])} & -- & -- & "
                     f"{co[e]['max_abs_err_vs_jax']:.0f}$^*$\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    save("tab_edge_full.tex", L)
    nums["edge"] = dict(coral=co, trt=trt)

# ================================================================ reduced precision (export_onnx.py closed-loop checks, parsed from its logs)
PREC = [("floor: BIS read as an integer", "floor: BIS read as an integer"), ("TensorRT fp32", "TensorRT fp32"), ("LiteRT fp32", "LiteRT fp32"),
        ("TensorRT fp16", "TensorRT fp16 (network fp16, recurrences fp32)"), ("LiteRT dynamic", "LiteRT dynamic (int8 weights)"),
        ("onnxruntime int8", "int8 dense layers, onnxruntime"), ("TensorRT int8", "int8 dense layers, TensorRT")]
logs = [os.path.join(DATA, f) for f in ("onnx_export.log", "onnx_export_litert.log") if os.path.exists(os.path.join(DATA, f))]
if logs:
    rows, npat = {}, None
    for f in logs:
        for l in open(f):
            m = re.match(r"closed loops, (\d+) patients", l)
            if m:
                npat = int(m.group(1))
            for key, _ in PREC:
                if l.strip().startswith(key):                  # the five numbers after the 42-character label column
                    rows[key] = [float(v) for v in re.findall(r"[-+]?\d+\.\d+(?:e[-+]\d+)?", l[42:])[:5]]
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"model & $\Delta S$ & $|\Delta u|$ p99 & max & $|\Delta\mathrm{BIS}|$ p99 & max\\", r"\midrule"]
    for key, nm in PREC:
        if key in rows:
            v = rows[key]
            L.append(f"{nm} & {f'{v[0]:+.1e}'.replace('-', '$-$', 1) if v[0] < 0 else f'{v[0]:+.1e}'} & "
                     f"{v[1]:.2g} & {v[2]:.3g} & {v[3]:.3f} & {v[4]:.2f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    save("tab_prec.tex", L)
    nums["prec"] = dict(rows=rows, patients=npat)

json.dump(nums, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
print("ok; MPC baselines complete:", nums["have_mpc"])
