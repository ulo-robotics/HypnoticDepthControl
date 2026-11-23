#!/usr/bin/env python
"""
Fine-tune the search's result on the training distribution with long, variable-length episodes (the last stage of the procedure).

The starting points are the search winner (--starts winner) or every final-stage model of the search (--starts finals: its best snapshots,
several configurations x training seeds; the seed decides much of how a model trades induction speed against overshoot under delay). Each
starting point continues (its EMA weights, a fresh optimizer state, the iteration count restarted) on model.TRAIN with signed multi-step
stimulation, episodes of 30 .. --minutes min, as variants of its configuration: the learning rate x --lr-mults, each with the wsd rule
decaying from --decay-at x --iters (100 warm-up iterations; no delay curriculum), and --w-low (the training weight below BIS 40) when given.
The variants train in populations of at most --pop (a population that does not fit in GPU memory is split). Every --eval-every iterations
each one's EMA is validated on new patients and its best snapshot kept. The best snapshots and the unchanged winner are re-ranked on a fresh
key, the best becomes the model, and it is tested ONCE on the search's test key against the winner and the untrained base prior.

  python finetune.py --search runs/search --out runs/finetune
  python finetune.py --search runs/search --out runs/finetune --starts finals --w-low 6 --lr-mults 1 --decay-at 0.333
  python finetune.py --search runs/search_smoke --out runs/finetune_smoke --smoke
"""
import argparse, json, math, os, pickle, time
import numpy as np
import jax
import jax.numpy as jnp

from lethe.core import model as M
from lethe.core import policy as P
from lethe.train import trainer as T
from lethe.train import search as S

LR_MULTS = (1.0, 1 / 3, 1 / 10, 1 / 30)
DECAY_AT = (1 / 3, 2 / 3)                     # wsd: constant learning rate until this share of --iters, then down to the floor in half that time


def variants(cfg, iters, lr_mults=LR_MULTS, decay_at=DECAY_AT, minutes=120.0):
    return [dict(cfg, stimulus="signed", len_lo=30.0, minutes=minutes, delay_ramp=0, lr=cfg["lr"] * m, sched="wsd", tc=d * iters, warm=100)
            for m in lr_mults for d in decay_at]


def starts_of(search, which):
    """the starting points: [(name, compact parameters, configuration)]"""
    if which == "winner":
        win = P.load_model(os.path.join(search, "best_model.pkl"))
        return [("winner", win["params"], win["cfg"])]
    out = []
    for t in json.load(open(os.path.join(search, "manifest.json")))["trials"]:
        if t.get("failed"):
            continue
        b, s_ = (os.path.join(search, "store", f"{x}{t['id']}.pkl") for x in ("b", "t"))
        st = pickle.load(open(b if os.path.exists(b) else s_, "rb"))
        out.append((f"config {t['parent']} seed {t['seed']}", P.shrink(st["ema"], t["cfg"]), t["cfg"]))
    return out


class Keep:
    """on_block of one population: validate its members, keep each one's best snapshot, log"""

    def __init__(self, pop, cfgs, best, log):
        self.pop, self.cfgs, self.best, self.log = pop, cfgs, best, log

    def __call__(self, it, m, dt):
        res = self.pop.validate(jax.random.fold_in(T.VAL_KEY, 50_000 + it))
        states = None
        for i, (_, sc) in enumerate(res):
            if sc < self.best[i]["score"]:
                states = states or self.pop.member_states()
                self.best[i] = dict(score=sc, it=it, params=P.shrink(states[i]["ema"], self.cfgs[i]))
        self.log.write(json.dumps(dict(it=it, scores=[sc for _, sc in res], train_J=[float(x) for x in m["J"]], time=time.time())) + "\n")
        self.log.flush()
        print(f"it {it:5d}  val " + " ".join(f"{sc:6.2f}" for _, sc in res) + f"  ({1000 * dt:.0f} ms/it)", flush=True)


def train_group(members, a, log):
    """train members [(cfg, compact start params)] to --iters; [best snapshot dicts]. Out of GPU memory: the group is split in two"""
    cfgs = [c for c, _ in members]
    try:
        with jax.default_device(T.HOST):
            states = []
            for c, cp in members:
                tp = T.to_np(P.expand(jax.tree_util.tree_map(jnp.asarray, cp))[0])
                states.append(dict(params=tp, ema=tp, ost={c["opt"]: T.to_np(T.OPTS[c["opt"]].init(tp))}, it=0))
        pop = T.Population(cfgs, states, jax.random.PRNGKey(20_000), val_batch=a.val_batch, iters_hint=a.iters, block=a.eval_every,
                           engine=a.engine, full_tune=a.full_tune)
        print(f"population of {len(cfgs)}, engine {pop.engine}", flush=True)
        best = [dict(score=math.inf, it=-1, params=None) for _ in cfgs]
        pop.train_to(a.iters, Keep(pop, cfgs, best, log))
        return best
    except Exception as e:
        if not S._oom(e) or len(members) == 1:
            raise
        print(f"out of memory with {len(members)} members: splitting", flush=True)
        h = len(members) // 2
        return train_group(members[:h], a, log) + train_group(members[h:], a, log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True, help="a finished search.py output folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-batch", type=int, default=1024)
    ap.add_argument("--rerank-batch", type=int, default=4096)
    ap.add_argument("--test-batch", type=int, default=4096)
    ap.add_argument("--lr-mults", default=",".join(map(str, LR_MULTS)), help="learning-rate multipliers of the variants (fewer: less GPU memory)")
    ap.add_argument("--decay-at", default=",".join(map(str, DECAY_AT)))
    ap.add_argument("--minutes", type=float, default=120.0, help="episode length (min) of training (episodes of 30 min .. this), validation and test")
    ap.add_argument("--starts", default="winner", choices=["winner", "finals"], help="fine-tune the search winner, or every final-stage model")
    ap.add_argument("--w-low", type=float, default=None, help="the training weight below BIS 40 (default: each configuration's own)")
    ap.add_argument("--pop", type=int, default=6, help="variants trained at once (at most)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    ap.add_argument("--engine", default="auto", choices=["auto", "raw", "adjoint", "cblk"], help="gradient engine (auto: tuned; see lethe/engine)")
    ap.add_argument("--full-tune", action="store_true", help="tune every engine candidate of a new shape")
    a = ap.parse_args()
    T.limit_malloc_arenas()
    if a.smoke:
        a.iters, a.eval_every, a.val_batch, a.rerank_batch, a.test_batch = 20, 10, 64, 64, 64
    if os.path.exists(os.path.join(a.out, "results.json")):
        raise SystemExit(f"{a.out} already holds a finished fine-tune")
    os.makedirs(a.out, exist_ok=True)

    win = P.load_model(os.path.join(a.search, "best_model.pkl"))
    sargs = json.load(open(os.path.join(a.search, "manifest.json")))["args"]
    lrs, decs = [float(x) for x in a.lr_mults.split(",")], [float(x) for x in a.decay_at.split(",")]
    members, names = [], []
    for sname, cp, scfg in starts_of(a.search, a.starts):
        for (m, d), c in zip([(m, d) for m in lrs for d in decs], variants(scfg, a.iters, lrs, decs, a.minutes)):
            c = dict(c, w_low=a.w_low) if a.w_low is not None else c
            members.append((dict(c, batch=32) if a.smoke else c, cp))
            names.append(f"{sname}, lr x{m:.3g}, decay at {d:.2f}")
    order = sorted(range(len(members)), key=lambda i: tuple(int(members[i][0][k]) for k in ("L", "H", "NS", "head")))   # shapes together
    members, names = [members[i] for i in order], [names[i] for i in order]
    print(f"fine-tuning {len(members)} variants ({a.starts}: {len({n.split(', lr')[0] for n in names})} starting points)"
          + (f", w_low {a.w_low:g}" if a.w_low is not None else ""), flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    best = []
    for g in range(0, len(members), a.pop):
        best += train_group(members[g:g + a.pop], a, log)
    print(f"trained in {(time.time() - t0) / 60:.1f} min", flush=True)
    cfgs = [c for c, _ in members]

    # re-rank the best snapshots and the winner itself on a fresh key; the best one is tested once
    names = names + ["search winner (not fine-tuned)"]
    models = [b["params"] for b in best] + [win["params"]]
    w = np.array([M.VAL[s][1] for s in M.VAL])
    comb = lambda st: sum(w[j] * (st[s]["J"] + 2.0 * st[s]["viol"]) for j, s in enumerate(M.VAL)) / w.sum()     # per patient
    sc = np.stack([comb(st) for st in T.evaluate(models, a.rerank_batch, jax.random.fold_in(S._keys()["rerank"], 1), per_patient=True,
                                                 minutes=a.minutes)])
    mean = sc.mean(1)
    bi = int(np.argmin(mean))
    print("\n=== re-rank (fresh key, %d patients per scenario) ===" % a.rerank_batch)
    for i in np.argsort(mean):
        print(f"  {names[i]:48s} {mean[i]:.3f}   (best validation {best[i]['score']:.3f} at {best[i]['it']})" if i < len(best)
              else f"  {names[i]:48s} {mean[i]:.3f}")

    _, BASE, _ = S.space()
    fix = {k: json.loads(v) for k, v in (x.split("=", 1) for x in sargs.get("fix") or [])}
    base = P.init_params(jax.random.PRNGKey(0), dict(BASE, **fix))
    test = np.stack([comb(st) for st in T.evaluate([models[bi], win["params"], base], a.test_batch, S._keys()["test"], per_patient=True,
                                                   minutes=a.minutes)])
    ci = lambda d: float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))
    cfg = cfgs[bi] if bi < len(cfgs) else dict(win["cfg"])
    res = dict(model=names[bi], cfg=cfg, iters=best[bi]["it"] if bi < len(best) else 0,
               rerank=[dict(name=n, mean=float(mean[i]), best_val=best[i]["score"] if i < len(best) else None,
                            best_it=best[i]["it"] if i < len(best) else None)
                       for i, n in enumerate(names)],
               test_score=float(test[0].mean()), winner_score=float(test[1].mean()), base_prior_score=float(test[2].mean()),
               test_improvement=float((test[2] - test[0]).mean()), test_improvement_ci=ci(test[2] - test[0]),
               gain_over_winner=float((test[1] - test[0]).mean()), gain_over_winner_ci=ci(test[1] - test[0]))
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    T.save(dict(params=models[bi], cfg=cfg), os.path.join(a.out, "best_model.pkl"))
    print(f"\nmodel: {names[bi]}; test score {res['test_score']:.3f}, search winner {res['winner_score']:.3f} "
          f"(gain {res['gain_over_winner']:.3f} +- {res['gain_over_winner_ci']:.3f}), untrained base prior {res['base_prior_score']:.3f} "
          f"(improvement {res['test_improvement']:.3f} +- {res['test_improvement_ci']:.3f})\n-> {os.path.join(a.out, 'best_model.pkl')}", flush=True)


if __name__ == "__main__":
    main()
