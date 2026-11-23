#!/usr/bin/env python
"""
Train the policy (one configuration) with pathwise gradients through the differentiable closed loop.

Every iteration draws a fresh batch of patients from model.TRAIN (a share of them exactly nominal; the monitor-delay range ramps up over the
first iterations), computes one gradient per patient (lethe/engine: the fastest verified engine, raw JAX as the fallback), clips every patient's
gradient to k x the median patient norm, averages, and takes an optimizer step (per-group learning rates for the physics layer, the SSM time
scales and the rest). An EMA of the weights is validated every --eval-every iterations on fresh patients from all validation scenarios; the
best EMA snapshot is written to <out>/best_model.pkl (compact layout). The run is a trainer.Population of one: the same program as a trial
of search.py.

  python train.py --out runs/train                       # defaults = the selected configuration (trainer.SELECTED)
  python train.py --out runs/train --resume              # continue after an interruption
  python train.py --out runs/smoke --iters 30 --batch 64 --val-batch 128 --eval-every 10 --H 16 --L 1 --head 0   # quick check
  python train.py --engine raw ...                       # force raw JAX autodiff (see lethe/engine)
"""
import argparse, json, os, pickle, signal, time
import jax

from lethe.core import policy as P
from lethe.train import trainer as T

CONFIG = dict(T.SELECTED, iters=6000, seed=0)
aggregate = T.aggregate                                    # (used by selftest.py)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-batch", type=int, default=1024, help="patients per validation scenario (new patients at every validation)")
    ap.add_argument("--ckpt-every", type=int, default=500)
    for k, v in CONFIG.items():
        ap.add_argument(f"--{k}", type=type(v), default=None, help=f"default {v}")
    ap.add_argument("--engine", default="auto", choices=["auto", "raw", "adjoint", "cblk"], help="gradient engine (auto: tuned; see lethe/engine)")
    ap.add_argument("--full-tune", action="store_true", help="tune every engine candidate of a new shape")
    args = ap.parse_args()
    T.limit_malloc_arenas()
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "checkpoint.pkl")

    over = {k: getattr(args, k) for k in CONFIG if getattr(args, k) is not None}
    if args.resume and os.path.exists(ckpt_path):
        ck = pickle.load(open(ckpt_path, "rb"))
        c = {**ck["cfg"], **{k: v for k, v in over.items() if k == "iters"}}      # only the length may change on resume
        state, best = ck["state"], ck["best"]
        print(f"resumed at iteration {state['it']} (best score {best['score']:.3f} at {best['it']})")
    else:
        if os.path.exists(ckpt_path):
            raise SystemExit(f"{ckpt_path} exists: pass --resume or use another --out")
        c = {**CONFIG, **over}
        state, best = T.new_state(c, jax.random.PRNGKey(c["seed"])), dict(score=float("inf"), it=-1)
    json.dump(c, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print(f"config: {json.dumps(c)}\nbackend {jax.default_backend()}", flush=True)
    pop = T.Population([c], [state], jax.random.PRNGKey(10_000 + c["seed"]), val_batch=args.val_batch, iters_hint=c["iters"] - state["it"],
                       block=args.eval_every, engine=args.engine, full_tune=args.full_tune)
    log = open(os.path.join(args.out, "log.jsonl"), "a")

    stop = []
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))

    def evaluate(it, m=None, dt=None):
        nonlocal best
        (st, sc), = pop.validate(jax.random.fold_in(T.VAL_KEY, it))
        if sc < best["score"]:
            best = dict(score=sc, it=it)
            ema = jax.tree_util.tree_map(lambda a: a[0], T.to_np(pop.state["ema"]))
            T.save(dict(params=P.shrink(ema, c), cfg=c, it=it, score=sc, val=st), os.path.join(args.out, "best_model.pkl"))
        train = None if m is None else {k: float(v[0]) for k, v in m.items()}
        log.write(json.dumps(dict(it=it, score=sc, val=st, train=train, time=time.time())) + "\n")
        log.flush()
        if train is None:
            print(f"it {it}  score {sc:.3f}", flush=True)
        else:
            print(f"it {it:5d}  loss {train['loss']:.4f}  train J {train['J']:7.2f}  clipped {100 * train['clipped']:4.1f}%  "
                  f"val score {sc:7.3f}  (best {best['score']:.3f} @ {best['it']})  {1000 * dt:.0f} ms/it", flush=True)
        if it % args.ckpt_every == 0 or it == c["iters"] or stop:
            T.save(dict(cfg=c, state=pop.member_states()[0], best=best), ckpt_path)

    if pop.it == 0:
        evaluate(0)
    pop.train_to(c["iters"], on_block=evaluate, stop=lambda: bool(stop))
    if stop:
        print(f"stopped at iteration {pop.it}; continue with --resume")
        raise SystemExit(130)
    print(f"done: best score {best['score']:.3f} at iteration {best['it']} -> {os.path.join(args.out, 'best_model.pkl')}")


if __name__ == "__main__":
    main()
