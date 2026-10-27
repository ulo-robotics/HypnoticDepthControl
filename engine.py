"""
Training engine selection: per-patient gradients of a POPULATION of networks (leading member axis on the parameters, sizes and loss weights;
the patients are shared), either by raw JAX autodiff (vmap of grad through policy.rollout; always available, the fallback and the reference)
or by the adjoint method of kstep.py, whose forward and backward sweeps each run on the fastest VERIFIED implementation:
  xla      lax.scan sweeps (kstep.xla_forward / xla_backward)           always available; per step (the reference) or with the S4D
                                                                         blocked over T steps (state traffic once per block)
  pallas   Pallas / Triton kernels (kpallas.py)                          GPU; some sizes do not fit (shared memory)
  cuda     hand-written CUDA kernels via XLA FFI (kcuda.py)              GPU + nvcc; one unpadded network (train.py)
Selection per shape (padded sizes, members, batch, device): every candidate is compiled, checked against the XLA reference on stable (nominal)
patients and timed on the real batch; the forward and the backward sweep are chosen independently. A candidate that fails to compile or
disagrees with the reference is never used. Raw JAX is the fallback when no adjoint candidate works (it was slower than the adjoint path at
every measured shape, and vmapped over a population it mostly does not fit in memory), and serves runs too short to pay for tuning.
Decisions are cached in $PKPD_CACHE/engine.json.

PKPD_ENGINE = auto (default) | raw | adjoint | pallas | cuda    (adjoint / pallas / cuda force that sweep backend; raw forces the fallback)
"""
import hashlib, json, os, time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import model as M
import policy as P
import kstep as K

CACHE = os.environ.get("PKPD_CACHE") or os.path.expanduser("~/.cache/pkpd_v1")
TUNE_FILE = os.path.join(CACHE, "engine.json")
MIN_TUNE_ITERS = 200             # shorter runs use raw JAX untuned (tuning compiles every candidate)
TOL = 1e-3                       # relative tolerance of a candidate's tapes vs the XLA reference (float32, stable patients)

XLA_CANDS = [dict(T=10), dict(T=20)]
PALLAS_CANDS = [dict(kb=16, nw=4, lazy=False), dict(kb=16, nw=8, lazy=False), dict(kb=16, nw=8, lazy=True)]
CUDA_CANDS = [dict(wpb=1, stage=0), dict(wpb=4, stage=0), dict(wpb=8, stage=1), dict(wpb=16, stage=1)]


def _code_hash():
    h = hashlib.sha1()
    d = os.path.dirname(os.path.abspath(__file__))
    for f in ("kstep.py", "kpallas.py", "kcuda.py", "cuda/sweep.cu", "policy.py", "model.py", "engine.py"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def _sig(tp, B, plain):
    s, dev = P.shape_of(tp), jax.devices()[0]
    n = jax.tree_util.tree_leaves(tp)[0].shape[0]
    return (f"H{s['H']}_NS{s['NS']}_L{s['L']}_hd{s['HD']}_P{s['P']}_x{n}_{'masked' if plain is None else 'plain' + str(int(plain))}_B{B}_"
            f"{dev.platform}_{getattr(dev, 'device_kind', '')}_{_code_hash()}").replace(" ", "")


def _load():
    try:
        return json.load(open(TUNE_FILE))
    except (OSError, ValueError):
        return {}


def _save(sig, rec):
    os.makedirs(CACHE, exist_ok=True)
    d = _load()
    d[sig] = rec
    json.dump(d, open(TUNE_FILE + ".tmp", "w"), indent=1)
    os.replace(TUNE_FILE + ".tmp", TUNE_FILE)


def pop_map(f, n, in_axes):
    """f over the member axis: vmap, or for a single member a plain call (no batching rule needed: the CUDA kernels have none)"""
    if n > 1:
        return jax.vmap(f, in_axes=in_axes)
    sq = lambda a, ax: a if ax is None else jax.tree_util.tree_map(lambda x: x[0], a)

    def g(*args):
        return jax.tree_util.tree_map(lambda x: x[None], f(*[sq(a, ax) for a, ax in zip(args, in_axes)]))
    return g


# ---------------------------------------------------------------- the sweep implementations
def sweep_fns(kind, opt):
    if kind == "xla":
        return partial(K.xla_forward, **opt), partial(K.xla_backward, **opt)
    if kind == "pallas":
        import kpallas
        return partial(kpallas.forward, **opt), partial(kpallas.backward, **opt)
    if kind == "cuda":
        import kcuda
        return partial(kcuda.forward, **opt), partial(kcuda.backward, **opt)
    raise ValueError(kind)


def _gpu():
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def candidates(B, n=1, plain=None, only=None):
    out = [("xla", {})] + [("xla", o) for o in XLA_CANDS]
    if _gpu():
        out += [("pallas", o) for o in PALLAS_CANDS if B % o["kb"] == 0]
        try:
            import kcuda
            if n == 1 and plain is not None and kcuda.available():
                out += [("cuda", o) for o in CUDA_CANDS if B % o["wpb"] == 0]
        except Exception:
            pass
    return [c for c in out if only is None or c[0] in (only, "xla")]


# ---------------------------------------------------------------- gradient functions: (tp, sz, pts, w_low, rho_u) -> per-patient grads, loss, J
def raw_grads(tp, sz, pts, w_low, rho_u):
    """one network, raw JAX"""
    def loss1(p, pt):
        one = {k: (v[:, :, None] if k == "z" else v[None]) for k, v in pt.items()}
        loss, b, _ = P.rollout(p, one, w_low, rho_u, sz=sz)
        return loss[0], jnp.mean((b[:, 0] - pt["tgt"]) ** 2)
    (loss, J), g = jax.vmap(jax.value_and_grad(loss1, has_aux=True), in_axes=(None, P.PATIENT_AXES))(tp, pts)
    return g, loss, J


def population(fn, n):
    """a one-network gradient function -> the population version (members along the leading axis, shared patients)"""
    return pop_map(fn, n, (0, 0, None, 0, 0))


def adjoint_fn(fwd, bwd, plain, chunk):
    return partial(K.adjoint_grads, plain=plain, forward=fwd, backward=bwd, chunk=chunk)


def _chunk(tp, n):
    """patients per chunk of the gradient assembly (its memory ~ members x chunk x H x NS x N complex values)"""
    s = P.shape_of(tp)
    return max(4, min(16, 1 << int(np.log2(max(1, 131072 // (s["H"] * s["NS"] * n))))))       # measured best: 8 to 16


# ---------------------------------------------------------------- selection
def _time(f, *a, reps=2):
    jax.block_until_ready(f(*a))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        best = min(best, time.perf_counter() - t0)
    return best


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if not np.all(np.isfinite(a)):
        return float("inf")
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


def _aot(fn, *args):
    """ahead-of-time compile (runs in a worker thread: XLA / Triton compilation releases the GIL)"""
    return jax.jit(fn).lower(*args).compile()


def select(tp, sz, B, w_low, rho_u, plain=None, iters=None, mode=None, log=print):
    """choose the gradient implementation for a population (tp, sz, w_low, rho_u with a leading member axis) on batches of B patients.
    Returns (grad_fn(tp, sz, pts, w_low, rho_u) -> (per-patient grads (n, B, ...), loss (n, B), J (n, B)), description).
    Every candidate sweep is compiled once, in parallel threads, then checked against the XLA reference on stable (nominal) patients and timed
    with that same executable (run time does not depend on the patients)."""
    from concurrent.futures import ThreadPoolExecutor
    mode = (mode or os.environ.get("PKPD_ENGINE") or "auto").lower()
    n = jax.tree_util.tree_leaves(tp)[0].shape[0]
    raw = population(raw_grads, n)
    if mode == "raw":
        return raw, dict(engine="raw")
    sig = _sig(tp, B, plain) + f"_{mode}"
    rec = _load().get(sig)
    chunk = _chunk(tp, n)
    if rec is None and mode == "auto" and iters is not None and iters < MIN_TUNE_ITERS:
        log(f"[engine] run too short to pay for tuning ({iters} its): raw JAX")
        return raw, dict(engine="raw", untuned=True)
    if rec is None:
        t0 = time.perf_counter()
        chk = M.sample_patients(jax.random.PRNGKey(4242), B, M.VAL["nominal"][0])
        kp = pop_map(lambda t, s, w, r: K.to_kp(t, s, w, r, plain), n, (0, 0, 0, 0))(tp, sz, w_low, rho_u)
        q, st = K.to_kq(chk)
        sweeps = lambda f, tape: pop_map(lambda k_, t_=None: f(k_, q, st) if t_ is None else f(k_, q, st, t_), n, (0,) if tape is None else (0, 0))
        ft_shape = jax.eval_shape(sweeps(K.xla_forward, None), kp)[0]
        bt_shape = jax.eval_shape(sweeps(K.xla_backward, 1), kp, ft_shape)
        only = None if mode == "auto" else ("xla" if mode == "adjoint" else mode)
        cands = candidates(B, n, plain, only)
        jobs = {}
        for kind, opt in cands:
            fwd, bwd = sweep_fns(kind, opt)
            jobs[(kind, repr(opt), "fwd")] = (sweeps(fwd, None), (kp,))
            jobs[(kind, repr(opt), "bwd")] = (sweeps(bwd, 1), (kp, ft_shape))
        asm = pop_map(lambda t, s, w, r, ft, bt: K.assemble(t, s, w, r, plain, q, st, ft, bt, chunk), n, (0, 0, 0, 0, 0, 0))
        jobs["asm"] = (asm, (tp, sz, w_low, rho_u, ft_shape, bt_shape))
        with ThreadPoolExecutor(int(os.environ.get("PKPD_COMPILE_THREADS", "") or 8)) as ex:
            fut = {k: ex.submit(_aot, f, *a) for k, (f, a) in jobs.items()}
            exe = {}
            for k, f in fut.items():
                try:
                    exe[k] = f.result()
                except Exception as e:                                  # compile failure, resources, missing toolchain: not a candidate
                    exe[k] = e
        t_compile = time.perf_counter() - t0

        def timed(k, *a):
            e = exe[k]
            if isinstance(e, Exception):
                raise e
            return _time(e, *a), e(*a)

        try:
            _, (ft, s) = timed(("xla", "{}", "fwd"), kp)
            _, bt = timed(("xla", "{}", "bwd"), kp, ft)
        except Exception as e:
            log(f"[engine] XLA reference sweeps unavailable ({type(e).__name__}): raw JAX")
            ft = None
        res = []
        if ft is not None:
            for kind, opt in cands:
                try:
                    tf, (ft2, s2) = timed((kind, repr(opt), "fwd"), kp)
                    tb, bt2 = timed((kind, repr(opt), "bwd"), kp, ft)
                except Exception as e:
                    log(f"[engine] {kind} {opt}: unavailable ({type(e).__name__}: {str(e).splitlines()[0][:110]})")
                    continue
                err_f = max([_rel(ft2["c"][k], ft["c"][k]) for k in ft["c"]] + [_rel(a, b) for a, b in zip(ft2["ys"], ft["ys"])] +
                            [_rel(s2["cost"], s["cost"])])
                err_b = max([_rel(bt2["lam"][k], bt["lam"][k]) for k in bt["lam"]] + [_rel(a, b) for a, b in zip(bt2["e"], bt["e"])])
                if not (err_f < TOL and err_b < TOL):
                    log(f"[engine] {kind} {opt}: REJECTED (error vs reference fwd {err_f:.1e}, bwd {err_b:.1e})")
                    continue
                log(f"[engine] {kind} {opt}: fwd {1000 * tf:.1f} ms, bwd {1000 * tb:.1f} ms (error fwd {err_f:.1e}, bwd {err_b:.1e})")
                res.append((kind, opt, (tf, tb)))
        if mode not in ("auto", "adjoint"):
            res = [x for x in res if x[0] == mode] or res
        rec = dict(engine="raw")
        if res:
            fb = min(res, key=lambda x: x[2][0])
            bb = min(res, key=lambda x: x[2][1])
            try:
                t_asm, _ = timed("asm", tp, sz, w_low, rho_u, ft, bt)
                rec = dict(engine="adjoint", fwd=[fb[0], fb[1]], bwd=[bb[0], bb[1]], fwd_ms=1000 * fb[2][0], bwd_ms=1000 * bb[2][1],
                           assemble_ms=1000 * t_asm, total_ms=1000 * (fb[2][0] + bb[2][1] + t_asm))
            except Exception as e:
                log(f"[engine] gradient assembly unavailable ({type(e).__name__}: {str(e).splitlines()[0][:100]}): raw JAX")
        rec["tune_s"] = time.perf_counter() - t0
        rec["compile_s"] = t_compile
        _save(sig, rec)
    if rec["engine"] == "raw":
        log("[engine] using raw JAX")
        return raw, rec
    log(f"[engine] using adjoint: forward {rec['fwd'][0]} {rec['fwd'][1]}, backward {rec['bwd'][0]} {rec['bwd'][1]}: "
        f"{rec['total_ms']:.0f} ms per gradient (tuning {rec['tune_s']:.0f} s)")
    return population(adjoint_fn(sweep_fns(*rec["fwd"])[0], sweep_fns(*rec["bwd"])[1], plain, chunk), n), rec


def forward_stats_fn(rec, plain=None):
    """one network: per-patient (J, viol) of a batch with the chosen forward sweep (validation); policy.rollout when the engine is raw"""
    if rec.get("engine") != "adjoint":
        def f(tp, sz, pts):
            _, b, _ = P.rollout(tp, pts, sz=sz)
            return jnp.mean((b - pts["tgt"].T) ** 2, 0), jnp.mean(jax.nn.relu(40 - b) ** 2, 0)
        return f
    fwd = sweep_fns(*rec["fwd"])[0]

    def f(tp, sz, pts):
        _, s = fwd(K.to_kp(tp, sz, 0.0, 0.0, plain), *K.to_kq(pts), tape=False)
        return s["se"] / M.N, s["viol"] / M.N
    return f
