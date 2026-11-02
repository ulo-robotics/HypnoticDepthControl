#!/usr/bin/env python
"""
Meta-optimization of the policy's training configuration (the procedure of the paper):

  1. prior screen    score --prior-screen prior designs (q_theta, sy_scale, rho_lqr, q_x) at iteration 0 (forward only: the untrained policy
                     IS the prior compensator) and keep the best --prior-keep share as the pool the trials draw their prior design from
  2. trials          --trials configurations: trial 0 = trainer.SEARCH_BASE, the rest a scrambled Sobol sequence over trainer.SPACE
  3. rungs           successive halving: train every trial to the rung budget (--rungs, cumulative iterations), score its EMA on new patients
                     (a fresh validation key per rung), keep the best 1/--eta (at least --final-top)
  4. final stage     the --final-top survivors continue for --final-iters with --final-seeds training-data seeds each (seed copies share the
                     trial's state at the start of the stage)
  5. re-rank / test  all final models on a fresh key with --rerank-batch patients per scenario, the configuration with the best mean over seeds
                     wins (its best seed is the model); the winner is scored ONCE on the test key against the untrained base prior
All trials see the same training patients (common random numbers). Score = weighted mean over model.VAL of J + 2 viol (lower is better).
Training and scoring use the fixed stimulus (--stimulus, model.STIM_FIXED); finetune.py then moves the winner to the random one.

Trials train in populations (trainer.Population): the trials of one network shape, up to --pop at a time, as one jitted program on the GPU,
each with its own hyperparameters and optimizer. Shapes are the trials' sizes with the head and the extra physics modes padded to the largest
values (cheap), and every population is padded to the full size of its shape (--pop, or the size an out-of-memory split left; copies of a
member, discarded; --no-pad-pop: up to the next power of two), so each shape has one program and one engine choice for the whole search. A
population that does not fit in memory is split in two. One process: JAX owns the GPU.
Everything is resumable (--resume) at population granularity: trial states live in <out>/store, the plan and all scores in <out>/manifest.json,
every score in log.jsonl.

  python search.py --out runs/search                                     # the paper's settings (defaults)
  python search.py --out runs/search --resume
  python search.py --out runs/search_smoke --smoke                         # minutes: tiny budgets, end to end
  python search.py summary --out runs/search                               # rung-0 marginals per categorical dimension
"""
import argparse, json, math, os, pickle, shutil, signal, sys, time
import numpy as np

SEED_STRIDE = 100_000                         # id of seed copy s of trial t = t + s * SEED_STRIDE
PAD_HEAD, PAD_EXTRA = 64, 16                  # every population carries the largest head and extra-mode count of the space
AHEAD = 2                                     # populations prepared ahead of the one training (a compile can outlast a small population)


def _keys():
    import jax
    import trainer as T
    return dict(val=T.VAL_KEY, rerank=jax.random.PRNGKey(424242), test=jax.random.PRNGKey(99999))


def screen(designs, n, stimulus):
    """iteration-0 scores of prior designs (vmapped over designs: only the physics layer differs, the residual output is zero)"""
    import jax, jax.numpy as jnp
    import model as M, policy as P, trainer as T
    base = dict(T.SEARCH_BASE, H=8, NS=4, L=1, head=0, n_extra=0)
    out = np.full(len(designs), np.inf)
    ok, phys = [], []
    for i, d in enumerate(designs):
        try:
            phys.append(P.init_params(jax.random.PRNGKey(0), {**base, **d})["phys"])
            ok.append(i)
        except Exception:                                         # e.g. the Riccati solver rejected the design
            pass
    params = P.init_params(jax.random.PRNGKey(0), base)
    names, specs = list(M.VAL), M.val_specs(stimulus)
    keys = _keys()

    stacked = {k: jnp.asarray([specs[s][k] for s in names], jnp.float32) for k in M.SPEC_NOMINAL}
    w = jnp.asarray([M.VAL[s][1] for s in names])

    @jax.jit
    def scores(phs):
        def per_scen(inp):                                         # one scenario's patients, every design at once (traced once)
            i, sp = inp
            pts = M.sample_patients(jax.random.fold_in(jax.random.fold_in(keys["val"], 999), i), n, sp)

            def one(ph):
                _, b, _ = P.rollout(dict(params, phys=ph), pts)
                return jnp.mean((b - pts["tgt"].T) ** 2) + 2.0 * jnp.mean(jax.nn.relu(40 - b) ** 2)
            return jax.vmap(one)(phs)
        per = jax.lax.map(per_scen, (jnp.arange(len(names)), stacked))     # (scenarios, designs)
        return (w[:, None] * per).sum(0) / w.sum()

    G = 32
    for g0 in range(0, len(ok), G):
        idx = ok[g0:g0 + G]
        chunk = phys[g0:g0 + G]
        chunk = chunk + [chunk[-1]] * (G - len(chunk))
        sc = np.asarray(scores(jax.tree_util.tree_map(lambda *a: jnp.stack(a), *chunk)))
        out[idx] = sc[:len(idx)]
    return [float(x) for x in out]


def space():
    import trainer as T
    return T.SPACE, T.SEARCH_BASE, T.PRIOR_KEYS


def sobol(n, d, seed):
    from scipy.stats import qmc
    m = int(math.ceil(math.log2(max(n, 2))))
    return qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)[:n]


def decode(u, SPACE, pool, fix):
    cfg = {}
    for x, (name, kind, a, b) in zip(u, SPACE):
        cfg[name] = float(math.exp(math.log(a) + x * (math.log(b) - math.log(a)))) if kind == "log" else a[min(int(x * len(a)), len(a) - 1)]
    if pool:
        i = [n for n, *_ in SPACE].index("q_theta")
        cfg.update(pool[min(int(u[i] * len(pool)), len(pool) - 1)])
    cfg.update(fix)
    return cfg


def shape_key(c):
    """padded (L, H, NS) of a configuration: H and NS rounded up to a few sizes, so that a search compiles few programs (at most 3 x 3 x 3)"""
    up = lambda v, sizes: next((s for s in sizes if s >= int(v)), int(v))
    return (int(c["L"]), up(c["H"], (16, 32, 64)), up(c["NS"], (8, 16, 32)))


def group_shape(g):
    """padded (L, H, NS) of a population: the largest of its members'"""
    return tuple(max(x) for x in zip(*(shape_key(t["cfg"]) for t in g)))


TIGHT = 1000                                  # iterations from which a small population is not padded to its shape's full size (Search.members)


def p2(n):
    return 1 << max(0, int(n) - 1).bit_length()


def _oom(e):
    return "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()


class Search:
    def __init__(self, args):
        self.a = args
        self.out = args.out
        self.store = os.path.join(self.out, "store")
        self.mpath = os.path.join(self.out, "manifest.json")
        os.makedirs(self.store, exist_ok=True)

    # ---- bookkeeping
    def save(self, **kw):
        self.man.update(kw)
        tmp = self.mpath + ".tmp"
        json.dump(self.man, open(tmp, "w"), indent=1, default=float)
        os.replace(tmp, self.mpath)

    def log(self, rec):
        with open(os.path.join(self.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    def spath(self, t):
        return os.path.join(self.store, f"t{t['id']}.pkl")

    # ---- phases
    def prior_screen(self, SPACE, BASE, PRIOR_KEYS):
        a = self.a
        rng = {n: (lo, hi) for n, k, lo, hi in SPACE if n in PRIOR_KEYS}
        pts = sobol(a.prior_screen, len(PRIOR_KEYS), a.seed + 1)
        designs = [{k: BASE[k] for k in PRIOR_KEYS}] + [
            {k: float(math.exp(math.log(rng[k][0]) + u * (math.log(rng[k][1]) - math.log(rng[k][0])))) for k, u in zip(PRIOR_KEYS, p)} for p in pts]
        t0 = time.time()
        scores = np.asarray(screen(designs, a.val_batch, a.stimulus))
        order = np.argsort(scores)
        keep = max(8, int(np.isfinite(scores).sum() * a.prior_keep))
        pool = [designs[i] for i in order[:keep]]
        print(f"prior screen: {len(designs)} designs in {time.time() - t0:.0f}s, init score best {np.min(scores):.1f} median "
              f"{np.median(scores[np.isfinite(scores)]):.1f}; base design rank {int(np.where(order == 0)[0][0]) + 1}; kept {keep}", flush=True)
        json.dump(dict(pool=pool, scores=[float(scores[i]) for i in order[:keep]]), open(os.path.join(self.out, "prior_pool.json"), "w"))
        return pool

    def cap(self, g):
        """the most trials a population of g's shape holds (--pop, or less after an out-of-memory split)"""
        return min(self.max_pop.get(str(group_shape(g)), self.a.pop), self.a.pop)

    def members(self, g, iters=0):
        """members of g's compiled population: padded to its shape's full size (one program per shape), or to the next power of two. A
        population at most half full that still trains TIGHT iterations or more is padded only to the next power of two of its own size:
        the padding's compute would outweigh a compile (and an engine check) of its own"""
        if self.a.pad_pop and not (iters >= TIGHT and 2 * len(g) <= self.cap(g)):
            return p2(self.cap(g))
        return p2(len(g))

    def populations(self, todo):
        """the trials in populations of up to --pop, in size order so that neighbours share a padded shape: populations stay full when the
        trials of each shape thin out in the later rungs (deterministic: a resumed run forms the same populations)"""
        out, g = [], []
        for t in sorted(todo, key=lambda t: (shape_key(t["cfg"]), int(t["cfg"]["H"]), int(t["cfg"]["NS"]), t["id"])):
            if g and len(g) + 1 > self.cap(g + [t]):
                out.append(g)
                g = []
            g.append(t)
        out += [g] if g else []
        return sorted(out, key=lambda g: -math.prod(group_shape(g)))                  # big ones first: an OOM shows early

    def train_group(self, trials, budget, val_idx, tag, train_seed=None):
        """train trials to `budget`, population by population. First every new (shape, size) has its gradient engine chosen (timed with the GPU
        to itself); then the populations train one after the other while helper threads build the next AHEAD ones on the host and compile
        their training block and validation (CPU work that does not wait for the GPU)."""
        from concurrent.futures import ThreadPoolExecutor
        todo = [t for t in trials if t.get("done") != tag and not t.get("failed")]
        queue = self.populations(todo)
        print(f"  [{tag}] {len(todo)} trials in {len(queue)} populations", flush=True)
        t0, seen, i = time.time(), set(), 0
        while i < len(queue):                                                         # engine choices (cached across rungs and runs)
            g = queue[i]
            key = (group_shape(g), self.members(g, budget - g[0].get("it", 0)))
            if key not in seen:
                try:
                    self.prepare(g, budget, train_seed, compile=False)
                except Exception as e:
                    if not _oom(e) or key[1] == 1:
                        raise
                    self.shrink(g, tag)
                    queue, seen, i = self.populations(todo), set(), 0                 # the shape's populations re-formed at the smaller size
                    continue
                seen.add(key)
            i += 1
        print(f"  [{tag}] engines for {len(seen)} (shape, population size) pairs: {time.time() - t0:.0f}s", flush=True)
        t0, done = time.time(), [0]
        ahead = lambda q: [ex.submit(self.prepare, g, budget, train_seed) for g in q[:AHEAD]]
        with ThreadPoolExecutor(AHEAD) as ex:
            nxt = ahead(queue)
            while queue:
                g = queue.pop(0)
                fut = nxt.pop(0)
                if len(queue) >= AHEAD:
                    nxt.append(ex.submit(self.prepare, queue[AHEAD - 1], budget, train_seed))
                try:
                    pop, t_prep = fut.result()
                    self.train_population(g, pop, t_prep, budget, val_idx, tag)
                except Exception as e:
                    if not _oom(e) or len(g) == 1:
                        raise
                    for f in nxt:
                        f.exception()                                                 # (discarded: the queue changes)
                    h = self.shrink(g, tag)
                    queue = [g[i:i + h] for i in range(0, len(g), h)] + queue
                    nxt = ahead(queue)
                    continue
                done[0] += len(g)
                el = time.time() - t0
                print(f"  [{tag}] {done[0]}/{len(todo)} trials, {el / 60:.1f} min, ETA {el / done[0] * (len(todo) - done[0]) / 60:.0f} min", flush=True)

    def shrink(self, g, tag):
        """g's shape did not fit: at most half its population size from now on (recorded: a resumed run keeps it). Returns the new cap"""
        key = str(group_shape(g))
        self.max_pop[key] = max(1, self.members(g) // 2)
        json.dump(self.max_pop, open(os.path.join(self.out, "max_pop.json"), "w"))
        print(f"  [{tag}] out of memory with {self.members(g)} members of shape {key}: splitting (at most {self.max_pop[key]} per population)",
              flush=True)
        return self.max_pop[key]

    def prepare(self, g, budget, train_seed, compile=True):
        """the population of trials g (states from the store, or new), with its gradient engine; compiled ahead of time unless compile=False"""
        import jax
        import trainer as T
        t0 = time.time()
        states = []
        for t in g:
            p = self.spath(t)
            states.append(pickle.load(open(p, "rb")) if os.path.exists(p) else T.new_state(t["cfg"], jax.random.PRNGKey(t["id"] % SEED_STRIDE)))
        n = self.members(g, budget - g[0].get("it", 0))
        cfgs = [t["cfg"] for t in g] + [g[0]["cfg"]] * (n - len(g))                  # copies of the first member fill the population
        states += [states[0]] * (n - len(g))
        L, H, NS = group_shape(g)
        tk = jax.random.PRNGKey(10_000) if train_seed is None else jax.random.fold_in(jax.random.PRNGKey(10_000), train_seed)
        pop = T.Population(cfgs, states, tk, shape=dict(H=H, NS=NS, L=L, HD=PAD_HEAD, P=5 + PAD_EXTRA), opt_names=T.OPT_NAMES,
                           val_batch=self.a.val_batch, iters_hint=budget - states[0]["it"], block=self.a.block, log=lambda m: None)
        if compile:
            pop.compile()
        return pop, time.time() - t0

    def train_population(self, g, pop, t_prep, budget, val_idx, tag):
        import jax
        import trainer as T
        t1, it0 = time.time(), pop.it

        def ckpt(it, m, dt):
            if it % 500 == 0 and it < budget:                                           # a crash loses at most 500 iterations
                for t, st in zip(g, pop.member_states()):
                    T.save(st, self.spath(t))
        pop.train_to(budget, on_block=ckpt)
        t2 = time.time()
        res = pop.validate(jax.random.fold_in(_keys()["val"], val_idx))
        for t, st, (per, sc) in zip(g, pop.member_states(), res):
            T.save(st, self.spath(t))
            failed = not np.isfinite(sc)
            t.update(it=st["it"], score=float(sc) if not failed else float("inf"), per=per, failed=failed, done=tag)
            self.log(dict(trial=t["id"], stage=tag, it=t["it"], score=t["score"], cfg=t["cfg"], per=per, engine=pop.engine.get("engine")))
        self.save()
        e = pop.engine
        eng = f"{e['fwd'][0]}/{e['bwd'][0]}" if e.get("engine") == "adjoint" else e.get("engine")
        s = pop.shape
        print(f"    {len(g)} trials L{s['L']} H{s['H']} NS{s['NS']} (x{pop.n}, {eng}): prepared in {t_prep:.0f}s, train {t2 - t1:.0f}s "
              f"({1000 * (t2 - t1) / max(1, budget - it0) / len(g):.1f} ms per trial-iteration), validation {time.time() - t2:.0f}s, "
              f"best score {min(t['score'] for t in g):.3f}", flush=True)

    def run(self):
        a = self.a
        SPACE, BASE, PRIOR_KEYS = space()
        fix = {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.fix or [])}
        rungs = [int(r) for r in a.rungs.split(",")]
        mp = os.path.join(self.out, "max_pop.json")
        self.max_pop = json.load(open(mp)) if os.path.exists(mp) else {}
        if a.resume and os.path.exists(self.mpath):
            self.man = json.load(open(self.mpath))
            print(f"resumed: phase {self.man['phase']}, rung {self.man.get('rung')}, {len(self.man['trials'])} trials", flush=True)
        else:
            if os.path.exists(self.mpath):
                raise SystemExit(f"{self.mpath} exists: pass --resume or use another --out")
            self.man = dict(args=vars(a), phase="rung", rung=0)
            pool = self.prior_screen(SPACE, BASE, PRIOR_KEYS) if a.prior_screen > 0 else None
            cfgs = [{**BASE, **fix}] + [{**BASE, **decode(u, SPACE, pool, fix)} for u in sobol(a.trials, len(SPACE), a.seed)][: a.trials - 1]
            for c in cfgs:
                c["batch"], c["stimulus"] = a.batch, a.stimulus
            self.save(trials=[dict(id=i, cfg=c, it=0, score=None, failed=False, parent=i, seed=0) for i, c in enumerate(cfgs)])
        trials = self.man["trials"]
        if self.man["phase"] == "done":
            res = json.load(open(os.path.join(self.out, "results.json")))
            print(f"this search is finished: winner trial {res['best_model']} (config {res['best_parent']}), test score {res['test_score']:.3f}; "
                  f"see results.json / best_model.pkl")
            return

        if self.man["phase"] == "rung":
            for r in range(self.man["rung"], len(rungs)):
                print(f"\n=== rung {r}: {len(trials)} trials to {rungs[r]} iterations ===", flush=True)
                self.train_group(trials, rungs[r], r, f"rung{r}")
                trials.sort(key=lambda t: t["score"])
                keep = a.final_top if r == len(rungs) - 1 else max(a.final_top, int(math.ceil(len(trials) / a.eta)))
                survivors = [t for t in trials[:keep] if not t.get("failed")]
                for t in trials:
                    if t not in survivors and os.path.exists(self.spath(t)):
                        os.remove(self.spath(t))
                for t in survivors[:5]:
                    c = t["cfg"]
                    print(f"  top: trial {t['id']:5d} score {t['score']:8.3f}  {c['opt']} {c['sched']} lr {c['lr']:.1e} H{c['H']} NS{c['NS']} L{c['L']} "
                          f"head{c['head']} +{c['n_extra']}m", flush=True)
                trials[:] = survivors
                self.save(rung=r + 1)
            # seed copies share the trial's state at the start of the final stage
            copies = []
            for s in range(1, a.final_seeds):
                for t in list(trials):
                    cid = t["id"] + s * SEED_STRIDE
                    shutil.copyfile(self.spath(t), os.path.join(self.store, f"t{cid}.pkl"))
                    copies.append(dict(t, id=cid, seed=s, parent=t["parent"], done=None))
            trials += copies
            self.save(phase="final", final_start=rungs[-1])

        if self.man["phase"] == "final":
            end = self.man["final_start"] + a.final_iters
            print(f"\n=== final stage: {len(trials)} models to {end} iterations ({a.final_seeds} data seeds) ===", flush=True)
            for s in range(a.final_seeds):
                self.train_group([t for t in trials if t["seed"] == s], end, 1000 + s, f"final-s{s}", train_seed=s)
            self.save(phase="rerank")
        self.finish(trials, BASE, fix)

    def finish(self, trials, BASE, fix):
        import jax
        import model as M, policy as P, trainer as T
        a = self.a
        live = [t for t in trials if not t.get("failed")]
        models = [P.shrink(pickle.load(open(self.spath(t), "rb"))["ema"], t["cfg"]) for t in live]
        w = np.array([M.VAL[s][1] for s in M.VAL])
        comb = lambda st: sum(w[j] * (st[s]["J"] + 2.0 * st[s]["viol"]) for j, s in enumerate(M.VAL)) / w.sum()     # per patient
        sc = np.stack([comb(st) for st in T.evaluate(models, a.rerank_batch, _keys()["rerank"], per_patient=True, stimulus=a.stimulus)])
        mean, se = sc.mean(1), sc.std(1, ddof=1) / math.sqrt(sc.shape[1])
        rows = []
        for p in sorted({t["parent"] for t in live}):
            idx = [i for i, t in enumerate(live) if t["parent"] == p]
            m = mean[idx]
            rows.append(dict(parent=p, n_seeds=len(idx), mean=float(m.mean()),
                             se=float(m.std(ddof=1) / math.sqrt(len(idx))) if len(idx) > 1 else float(se[idx[0]]), best=idx[int(np.argmin(m))]))
        bi = min(rows, key=lambda r: r["mean"])["best"]
        best = live[bi]
        print("\n=== re-rank (fresh key, %d patients per scenario) ===" % a.rerank_batch)
        for r in sorted(rows, key=lambda r: r["mean"]):
            print(f"  config {r['parent']:5d}: {r['mean']:.3f} +- {r['se']:.3f} over {r['n_seeds']} seeds")
        test = np.stack([comb(st) for st in T.evaluate([models[bi], P.init_params(jax.random.PRNGKey(0), dict(BASE, **fix))], a.test_batch,
                                                        _keys()["test"], per_patient=True, stimulus=a.stimulus)])
        d = test[1] - test[0]
        res = dict(best_model=best["id"], best_parent=best["parent"], cfg=best["cfg"], iters=best["it"],
                   rerank=[dict(id=t["id"], parent=t["parent"], seed=t["seed"], mean=float(mean[i]), se=float(se[i]),
                                diff_vs_best=float((sc[i] - sc[bi]).mean())) for i, t in enumerate(live)], configs=rows,
                   test_score=float(test[0].mean()), base_prior_score=float(test[1].mean()),
                   test_improvement=float(d.mean()), test_improvement_ci=float(1.96 * d.std(ddof=1) / math.sqrt(len(d))))
        json.dump(res, open(os.path.join(self.out, "results.json"), "w"), indent=1, default=float)
        T.save(dict(params=models[bi], cfg=best["cfg"]), os.path.join(self.out, "best_model.pkl"))
        self.save(phase="done")
        print(f"\nwinner: trial {best['id']} (config {best['parent']}), test score {res['test_score']:.3f}, untrained base prior "
              f"{res['base_prior_score']:.3f} (improvement {res['test_improvement']:.3f} +- {res['test_improvement_ci']:.3f})\n"
              f"-> {os.path.join(self.out, 'best_model.pkl')} and results.json", flush=True)
        summary(self.out)


def summary(out):
    """rung-0 marginals (every trial had the same budget there, so the comparison has no survivorship bias)"""
    recs = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
    r0 = [r for r in recs if r.get("stage") == "rung0" and r.get("score") is not None and np.isfinite(r["score"])]
    if not r0:
        return
    print(f"\nrung-0 marginals ({len(r0)} trials): median score per value")
    for dim in ("opt", "sched", "warm", "ema", "clip", "H", "NS", "L", "head", "n_extra"):
        vals = sorted({r["cfg"][dim] for r in r0}, key=str)
        print(f"  {dim:8s} " + "  ".join(f"{v}: {np.median([r['score'] for r in r0 if r['cfg'][dim] == v]):.2f}" for v in vals))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        ap = argparse.ArgumentParser()
        ap.add_argument("mode")
        ap.add_argument("--out", required=True)
        summary(ap.parse_args().out)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--trials", type=int, default=1025)
    ap.add_argument("--rungs", default="300,900,2700", help="cumulative iteration budgets of the successive-halving rungs")
    ap.add_argument("--eta", type=int, default=3)
    ap.add_argument("--final-top", type=int, default=3)
    ap.add_argument("--final-iters", type=int, default=3000)
    ap.add_argument("--final-seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512, help="training patients per iteration")
    ap.add_argument("--stimulus", default="fixed", choices=["fixed", "random"], help="stimulus of training and scoring (model.STIM_FIXED: "
                    "the paper's search; finetune.py then moves the winner to the random one)")
    ap.add_argument("--val-batch", type=int, default=1024, help="patients per validation scenario (rung scores)")
    ap.add_argument("--rerank-batch", type=int, default=4096)
    ap.add_argument("--test-batch", type=int, default=4096)
    ap.add_argument("--prior-screen", type=int, default=4096, help="prior designs scored at iteration 0 (0 = off)")
    ap.add_argument("--prior-keep", type=float, default=0.25)
    ap.add_argument("--pop", type=int, default=8, help="trials per population (at most; a population that does not fit is split)")
    ap.add_argument("--block", type=int, default=100, help="training iterations per jitted call")
    ap.add_argument("--no-pad-pop", dest="pad_pop", action="store_false", help="pad a population only to the next power of two (fewer discarded "
                    "copies in a rung's last population, but one more program and engine choice per population size)")
    ap.add_argument("--seed", type=int, default=0, help="Sobol scrambling seed")
    ap.add_argument("--fix", action="append", help="pin a dimension: --fix H=64 --fix opt='\"adam\"'")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (overrides the budgets)")
    args = ap.parse_args()
    if args.smoke:
        args.trials, args.rungs, args.final_iters, args.final_seeds, args.batch, args.block = 6, "20,40", 20, 2, 32, 10
        args.val_batch, args.rerank_batch, args.test_batch, args.prior_screen = 64, 64, 64, 16
        args.fix = (args.fix or []) + ["H=16", "NS=8", "L=1"]
    os.makedirs(args.out, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))     # docker stop / kill: same as Ctrl-C
    try:
        Search(args).run()
    except KeyboardInterrupt:
        print("interrupted: continue with --resume", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
