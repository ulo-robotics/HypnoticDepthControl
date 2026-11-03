#!/usr/bin/env python
"""
Fine-tune the search winner on the random stimulus (the last stage of the paper's procedure).

search.py trains and scores its trials with the fixed stimulus (model.STIM_FIXED: +0.3 in log C50 after 20 min), whose timing a policy can
learn to anticipate. This stage continues the winner (its EMA weights, a fresh optimizer state, the iteration count restarted) on model.TRAIN
with the random stimulus (every stimulated patient its own size and time), as one population of variants of its configuration: the learning
rate x {1, 1/3, 1/10, 1/30}, each with the wsd rule decaying from --iters/3 or from 2 --iters/3 (100 warm-up iterations; no delay
curriculum, the winner has seen every delay). Every --eval-every iterations each variant's EMA is validated on new patients (random
stimulus) and its best snapshot kept. The best snapshots and the unchanged winner are re-ranked on a fresh key, the best becomes the model,
and it is tested ONCE on the search's test key (random stimulus) against the winner and the untrained base prior.

  python finetune.py --search runs/search --out runs/finetune
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
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    a = ap.parse_args()
    if a.smoke:
        a.iters, a.eval_every, a.val_batch, a.rerank_batch, a.test_batch = 20, 10, 64, 64, 64
    if os.path.exists(os.path.join(a.out, "results.json")):
        raise SystemExit(f"{a.out} already holds a finished fine-tune")
    os.makedirs(a.out, exist_ok=True)

    win = P.load_model(os.path.join(a.search, "best_model.pkl"))
    sargs = json.load(open(os.path.join(a.search, "manifest.json")))["args"]
    cfgs = variants(win["cfg"], a.iters, [float(x) for x in a.lr_mults.split(",")], [float(x) for x in a.decay_at.split(",")], a.minutes)
    if a.smoke:
        cfgs = [dict(c, batch=32) for c in cfgs]
    with jax.default_device(T.HOST):
        tp, _ = P.expand(jax.tree_util.tree_map(jnp.asarray, win["params"]))
        tp = T.to_np(tp)
        states = [dict(params=tp, ema=tp, ost={c["opt"]: T.to_np(T.OPTS[c["opt"]].init(tp))}, it=0) for c in cfgs]
    pop = T.Population(cfgs, states, jax.random.PRNGKey(20_000), val_batch=a.val_batch, iters_hint=a.iters, block=a.eval_every)
    print(f"fine-tuning {len(cfgs)} variants of the search winner (config {win['cfg'].get('opt')} lr {win['cfg']['lr']:.1e}, "
          f"H{win['cfg']['H']} NS{win['cfg']['NS']} L{win['cfg']['L']}) on the random stimulus, engine {pop.engine}", flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    best = [dict(score=math.inf, it=-1, params=None) for _ in cfgs]

    def on_block(it, m, dt):
        res = pop.validate(jax.random.fold_in(T.VAL_KEY, 50_000 + it))
        states = None
        for i, (st, sc) in enumerate(res):
            if sc < best[i]["score"]:
                states = states or pop.member_states()
                best[i] = dict(score=sc, it=it, params=P.shrink(states[i]["ema"], cfgs[i]))
        log.write(json.dumps(dict(it=it, scores=[sc for _, sc in res], train_J=[float(x) for x in m["J"]], time=time.time())) + "\n")
        log.flush()
        print(f"it {it:5d}  val " + " ".join(f"{sc:6.2f}" for _, sc in res) + f"  ({1000 * dt:.0f} ms/it)", flush=True)

    t0 = time.time()
    pop.train_to(a.iters, on_block)
    print(f"trained in {(time.time() - t0) / 60:.1f} min", flush=True)

    # re-rank the best snapshots and the winner itself on a fresh key; the best one is tested once
    names = [f"lr x{m:.3g}, decay at {d:.2f}" for m in (float(x) for x in a.lr_mults.split(",")) for d in (float(x) for x in a.decay_at.split(","))] + ["search winner (not fine-tuned)"]
    models = [b["params"] for b in best] + [win["params"]]
    w = np.array([M.VAL[s][1] for s in M.VAL])
    comb = lambda st: sum(w[j] * (st[s]["J"] + 2.0 * st[s]["viol"]) for j, s in enumerate(M.VAL)) / w.sum()     # per patient
    sc = np.stack([comb(st) for st in T.evaluate(models, a.rerank_batch, jax.random.fold_in(S._keys()["rerank"], 1), per_patient=True, minutes=a.minutes)])
    mean = sc.mean(1)
    bi = int(np.argmin(mean))
    print("\n=== re-rank (fresh key, random stimulus, %d patients per scenario) ===" % a.rerank_batch)
    for i in np.argsort(mean):
        print(f"  {names[i]:32s} {mean[i]:.3f}   (best validation {best[i]['score']:.3f} at {best[i]['it']})" if i < len(best)
              else f"  {names[i]:32s} {mean[i]:.3f}")

    _, BASE, _ = S.space()
    fix = {k: json.loads(v) for k, v in (x.split("=", 1) for x in sargs.get("fix") or [])}
    base = P.init_params(jax.random.PRNGKey(0), dict(BASE, **fix))
    test = np.stack([comb(st) for st in T.evaluate([models[bi], win["params"], base], a.test_batch, S._keys()["test"], per_patient=True, minutes=a.minutes)])
    ci = lambda d: float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))
    cfg = cfgs[bi] if bi < len(cfgs) else dict(win["cfg"])
    res = dict(model=names[bi], cfg=cfg, iters=best[bi]["it"] if bi < len(best) else 0,
               rerank=[dict(name=n, mean=float(mean[i]), best_val=best[i]["score"] if i < len(best) else None, best_it=best[i]["it"] if i < len(best) else None)
                       for i, n in enumerate(names)],
               test_score=float(test[0].mean()), winner_score=float(test[1].mean()), base_prior_score=float(test[2].mean()),
               test_improvement=float((test[2] - test[0]).mean()), test_improvement_ci=ci(test[2] - test[0]),
               gain_over_winner=float((test[1] - test[0]).mean()), gain_over_winner_ci=ci(test[1] - test[0]))
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    T.save(dict(params=models[bi], cfg=cfg), os.path.join(a.out, "best_model.pkl"))
    print(f"\nmodel: {names[bi]}; test score (random stimulus) {res['test_score']:.3f}, search winner {res['winner_score']:.3f} "
          f"(gain {res['gain_over_winner']:.3f} +- {res['gain_over_winner_ci']:.3f}), untrained base prior {res['base_prior_score']:.3f} "
          f"(improvement {res['test_improvement']:.3f} +- {res['test_improvement_ci']:.3f})\n-> {os.path.join(a.out, 'best_model.pkl')}", flush=True)


if __name__ == "__main__":
    main()
