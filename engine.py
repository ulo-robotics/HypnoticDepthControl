"""
Training engine selection: per-patient gradients of a POPULATION of networks (leading member axis on the parameters, sizes and loss weights;
the patients are shared), either by raw JAX autodiff (vmap of grad through policy.rollout; always available, the fallback and the reference)
or by the adjoint method of kstep.py, whose forward and backward sweeps each run on the fastest VERIFIED implementation:
  xla      lax.scan sweeps (kstep.xla_forward / xla_backward)           always available; per step (the reference) or with the S4D
                                                                         blocked over T steps (state traffic once per block)
  pallas   Pallas / Triton kernels (kpallas.py)                          GPU; some sizes do not fit (shared memory)
  cuda     hand-written CUDA kernels via XLA FFI (kcuda.py)              GPU + nvcc; one unpadded network (train.py)
  cblk     CUDA kernels in the blocked layout (cbwd.py: cuda/cfwd.cu,    GPU + nvcc; populations, masked and plain layouts
           cuda/cbwd.cu): PB patients per thread block, GLU / head on tensor cores, S4D state touched once per T-step block
and the gradient assembly on one of
  xla      kstep.assemble (replayed steps' VJP; S4D kernels by FFT cross-correlation)      always available, the reference
  xla+s4   the same with the S4D-kernel gradients from cuda/s4grad.cu                     GPU + nvcc
  cuda     cuda/asm.cu (fused step VJP) + batched GEMMs + cuda/s4grad.cu (kcuda.assemble)  GPU + nvcc; masked and plain layouts
  tapes    joint backward + assembly (cbwd.grads): the cuda/cbwd.cu backward also writes the per-step cotangents, so the assembly is GEMMs and
           cuda/s4grad.cu (or all of it in one pass over the tapes, cuda/casm.cu) without replaying the steps; replaces the backward sweep
           (chosen when forward + joint < forward + backward + assembly)
Selection per shape (padded sizes, members, batch, device): every candidate is compiled, checked against the XLA reference on stable (nominal)
patients and timed on the real batch; the forward sweep, the backward sweep and the assembly are chosen independently. A candidate that fails
to compile or disagrees with the reference is never used. Raw JAX is the fallback when no adjoint candidate works (it was slower than the adjoint path at
every measured shape, and vmapped over a population it mostly does not fit in memory), and serves runs too short to pay for tuning.
Decisions are cached in $PKPD_CACHE/engine.json.

PKPD_ENGINE = auto (default) | raw | adjoint | pallas | cuda | cblk   (adjoint / pallas / cuda / cblk force that sweep backend; raw forces
the fallback). PKPD_TUNE=full tunes every candidate (default: a shape is tuned once, the preferred candidates first; see select).
PKPD_TUNE_DEBUG=1 logs every candidate's memory need against the free GPU memory.
"""
import gc, hashlib, json, os, re, time
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
FIT = 0.7                        # a candidate runs only if it needs less than this share of the free GPU memory: an execution that runs out of
                                 # memory leaks its allocation for the rest of the process (XLA runtime), and the free memory is fragmented
                                 # (an 8.4 GiB candidate failed with 11.7 GiB free)
FINAL, NFINAL = 1.15, 2          # stage winners within this factor of the best (at most NFINAL each) are timed as whole gradients

XLA_CANDS = [dict(T=10), dict(T=20)]
PALLAS_CANDS = [dict(kb=16, nw=4, lazy=False), dict(kb=16, nw=8, lazy=False), dict(kb=16, nw=8, lazy=True)]
CUDA_CANDS = [dict(wpb=1, stage=0), dict(wpb=4, stage=0), dict(wpb=8, stage=1), dict(wpb=16, stage=1)]
ASM_CUDA_CANDS = [dict(wpb=1, nseg=12), dict(wpb=2, nseg=12), dict(wpb=4, nseg=24)]
CBLK_CANDS = [dict(T=10, pb=8, lanes=32), dict(T=5, pb=8, lanes=16), dict(T=10, pb=8, lanes=16), dict(T=10, pb=16, lanes=32), dict(T=5, pb=32, lanes=16),
              dict(T=10, pb=32, lanes=16)]                                   # 8 patients per block for the wide networks, more for the narrow
TAPES_CANDS = [dict(T=10, pb=16, lanes=32), dict(T=10, pb=32, lanes=16)] + [dict(o, casm=True) for o in (
    dict(T=5, pb=8, lanes=32), dict(T=5, pb=8, lanes=16), dict(T=5, pb=16, lanes=16), dict(T=10, pb=16, lanes=32), dict(T=5, pb=32, lanes=16),
    dict(T=10, pb=32, lanes=16))]


def _code_hash():
    h = hashlib.sha1()
    d = os.path.dirname(os.path.abspath(__file__))
    for f in ("kstep.py", "kpallas.py", "kcuda.py", "cbwd.py", "cuda/sweep.cu", "cuda/asm.cu", "cuda/s4grad.cu", "cuda/cbwd.cu", "cuda/cfwd.cu", "cuda/casm.cu",
              "policy.py", "model.py", "engine.py"):
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
    if kind == "cblk":
        import cbwd
        return partial(cbwd.forward, **opt), partial(cbwd.backward, **opt)
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
            if kcuda.available():
                out += [("cblk", o) for o in CBLK_CANDS if B % o["pb"] == 0]
        except Exception:
            pass
    return [c for c in out if only is None or c[0] in (only, "xla")]


def asm_candidates():
    out = [("xla", {})]
    if _gpu():
        try:
            import kcuda
            if kcuda.available():
                out += [("xla", dict(s4="cuda"))] + [("cuda", o) for o in ASM_CUDA_CANDS] + [("tapes", o) for o in TAPES_CANDS]
        except Exception:
            pass
    return out


def asm_fn(kind, opt):
    """the gradient assembly: (tp, sz, w_low, rho_u, plain, q, st, ftape, btape, chunk) -> per-patient gradients"""
    if kind == "xla":
        if opt.get("s4") == "cuda":
            import kcuda
            return partial(K.assemble, s4=kcuda.s4d_grads)
        return K.assemble
    if kind == "cuda":
        import kcuda
        return lambda tp, sz, w, r, plain, q, st, ft, bt, chunk: kcuda.assemble(tp, sz, w, r, plain, q, st, ft, bt, None, **opt)
    if kind == "tapes":                                  # joint: ignores bt (runs its own backward)
        import cbwd
        return lambda tp, sz, w, r, plain, q, st, ft, bt, chunk: cbwd.grads(tp, sz, w, r, plain, q, st, ft, chunk=chunk, **opt)
    raise ValueError(kind)


def _no_backward(kp, q, st, ftape):
    return {}


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


def adjoint_fn(fwd, bwd, plain, chunk, asm=None):
    return partial(K.adjoint_grads, plain=plain, forward=fwd, backward=bwd, chunk=chunk, assembly=asm)


def _chunk(tp, n):
    """patients per chunk of the gradient assembly (its memory ~ members x chunk x H x NS x N complex values)"""
    s = P.shape_of(tp)
    return max(4, min(16, 1 << int(np.log2(max(1, 131072 // (s["H"] * s["NS"] * n))))))       # measured best: 8 to 16


# ---------------------------------------------------------------- selection
def _time(f, *a, reps=3, warm=1.0):
    """best of reps calls after at least warm s of calls (the GPU clocks ramp up over a few seconds: a candidate timed cold loses)"""
    t0 = time.perf_counter()
    jax.block_until_ready(f(*a))
    while time.perf_counter() - t0 < warm:
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


def _rel_tree(a, b):
    """largest per-leaf relative error (each leaf against its own scale, floored at 1e-6 of the largest leaf)"""
    la = [np.asarray(x, np.float64) for x in jax.tree_util.tree_leaves(a)]
    lb = [np.asarray(x, np.float64) for x in jax.tree_util.tree_leaves(b)]
    if not all(np.all(np.isfinite(x)) for x in la):
        return float("inf")
    top = max(float(np.max(np.abs(y))) for y in lb) + 1e-30
    return max(float(np.max(np.abs(x - y))) / (float(np.max(np.abs(y))) + 1e-6 * top) for x, y in zip(la, lb))


def _aot(fn, *args):
    """ahead-of-time compile (runs in a worker thread: XLA / Triton compilation releases the GIL)"""
    return jax.jit(fn).lower(*args).compile()


def _bare(e):
    """an exception without its traceback: a stored or re-raised failure must not keep the tuning frame (its executables and multi-GiB
    tapes) alive through a reference cycle"""
    return RuntimeError(f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")


def _need(exe):
    """bytes a compiled candidate allocates when it runs (outputs + scratch), None if unknown"""
    try:
        m = exe.memory_analysis()
        return m.output_size_in_bytes + m.temp_size_in_bytes - m.alias_size_in_bytes
    except Exception:
        return None


def _free():
    try:
        st = jax.devices()[0].memory_stats()
        return st["bytes_limit"] - st["bytes_in_use"]
    except Exception:
        return None


def _other_size(sig):
    """a decision cached for the same shape at another population size (the sizes of a search share their engine)"""
    size = lambda k: int(re.search(r"_x(\d+)_", k).group(1))
    shape = lambda k: re.sub(r"_x\d+_", "_x_", k, count=1)
    hits = [(k, v) for k, v in _load().items() if k != sig and shape(k) == shape(sig) and v.get("engine") == "adjoint"]
    return max(hits, key=lambda kv: size(kv[0]))[1] if hits else None


def _preferred(cands, acands):
    """the candidates that won at every measured shape on CUDA GPUs (blocked CUDA sweeps, CUDA tape assembly), with the XLA references"""
    c = [x for x in cands if x[0] == "cblk" or (x[0] == "xla" and not x[1])]
    a = [x for x in acands if (x[0] == "tapes" and x[1].get("casm")) or (x[0] == "xla" and not x[1])]
    return (c, a) if len(c) > 1 and len(a) > 1 else None


def _tune(tp, sz, B, w_low, rho_u, plain, n, chunk, cands, acands, mode, log):
    """compile every candidate (in parallel threads), check it against the XLA reference on stable (nominal) patients, time it with that same
    executable (run time does not depend on the patients). Returns the decision record (engine raw when no adjoint path works)"""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.perf_counter()
    chk = M.sample_patients(jax.random.PRNGKey(4242), B, M.VAL["nominal"][0])
    kp = pop_map(lambda t, s, w, r: K.to_kp(t, s, w, r, plain), n, (0, 0, 0, 0))(tp, sz, w_low, rho_u)
    q, st = K.to_kq(chk)
    sweeps = lambda f, tape: pop_map(lambda k_, t_=None: f(k_, q, st) if t_ is None else f(k_, q, st, t_), n, (0,) if tape is None else (0, 0))
    ft_shape = jax.eval_shape(sweeps(K.xla_forward, None), kp)[0]
    bt_shape = jax.eval_shape(sweeps(K.xla_backward, 1), kp, ft_shape)
    jobs = {}
    for kind, opt in cands:
        fwd, bwd = sweep_fns(kind, opt)
        jobs[(kind, repr(opt), "fwd")] = (sweeps(fwd, None), (kp,))
        jobs[(kind, repr(opt), "bwd")] = (sweeps(bwd, 1), (kp, ft_shape))
    for kind, opt in acands:
        af = asm_fn(kind, opt)
        jobs[("asm", kind, repr(opt))] = (pop_map(lambda t, s, w, r, ft, bt, af=af: af(t, s, w, r, plain, q, st, ft, bt, chunk), n, (0,) * 6),
                                          (tp, sz, w_low, rho_u, ft_shape, bt_shape))
    with ThreadPoolExecutor(int(os.environ.get("PKPD_COMPILE_THREADS", "") or 3)) as ex:   # 8 ran the host out of memory
        fut = {k: ex.submit(_aot, f, *a) for k, (f, a) in jobs.items()}
        exe = {}
        for k, f in fut.items():
            try:
                exe[k] = f.result()
            except Exception as e:                                  # compile failure, resources, missing toolchain: not a candidate
                exe[k] = _bare(e)
    t_compile = time.perf_counter() - t0

    def timed(k, *a):
        e = exe[k]
        if isinstance(e, Exception):
            raise e
        need, free = _need(e), _free()
        if os.environ.get("PKPD_TUNE_DEBUG"):
            st = jax.devices()[0].memory_stats() or {}
            log(f"[engine] {k}: needs {(need or 0) / 2**30:.2f} GiB, free {(free or 0) / 2**30:.2f} GiB, largest free block "
                f"{st.get('largest_free_block_bytes', 0) / 2**30:.2f} GiB")
        if need is not None and free is not None and need > FIT * free:   # an out-of-memory run leaks its allocation (XLA runtime): never try
            raise RuntimeError(f"needs {need / 2**30:.2f} GiB of {free / 2**30:.2f} GiB free: not run")
        try:
            return _time(e, *a), e(*a)
        except Exception as err:                                    # raised without the frames that hold the buffers
            raise _bare(err) from None

    ft = bt = None
    try:
        _, (ft, s) = timed(("xla", "{}", "fwd"), kp)
        _, bt = timed(("xla", "{}", "bwd"), kp, ft)
    except Exception as e:
        log(f"[engine] XLA reference sweeps unavailable ({type(e).__name__}): raw JAX")
        ft = None
    res = []
    if ft is not None:
        for kind, opt in cands:                        # one candidate's outputs at a time (checked, then released)
            try:
                tf, (ft2, s2) = timed((kind, repr(opt), "fwd"), kp)
                err_f = max([_rel(ft2["c"][k], ft["c"][k]) for k in ft["c"]] + [_rel(a, b) for a, b in zip(ft2["ys"], ft["ys"])] +
                            [_rel(s2["cost"], s["cost"])])
                del ft2, s2
                tb, bt2 = timed((kind, repr(opt), "bwd"), kp, ft)
                err_b = max([_rel(bt2["lam"][k], bt["lam"][k]) for k in bt["lam"]] + [_rel(a, b) for a, b in zip(bt2["e"], bt["e"])])
                del bt2
            except Exception as e:
                log(f"[engine] {kind} {opt}: unavailable ({type(e).__name__}: {str(e).splitlines()[0][:110]})")
                continue
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
        ares, g_ref = [], None
        for kind, opt in acands:
            try:
                ta, g = timed(("asm", kind, repr(opt)), tp, sz, w_low, rho_u, ft, bt)
            except Exception as e:
                log(f"[engine] assembly {kind} {opt}: unavailable ({type(e).__name__}: {str(e).splitlines()[0][:110]})")
                continue
            if g_ref is None:
                if kind != "xla" or opt:
                    break                                  # no reference: nothing else is trusted
                g_ref = jax.device_get(g)                  # the reference on the host: the candidates get the GPU memory
            err = _rel_tree(g, g_ref)
            del g
            if not err < TOL:
                log(f"[engine] assembly {kind} {opt}: REJECTED (error vs reference {err:.1e})")
                continue
            log(f"[engine] assembly {kind} {opt}: {1000 * ta:.1f} ms (error {err:.1e})")
            ares.append((kind, opt, ta))
        if ares:                                       # a joint candidate includes its backward: compare whole gradients
            total = lambda x: fb[2][0] + x[2] + (0.0 if x[0] == "tapes" else bb[2][1])
            ab = min(ares, key=total)
            fin_f = [x for x in sorted(res, key=lambda x: x[2][0]) if x[2][0] <= FINAL * fb[2][0]][:NFINAL]
            fin_a = [x for x in sorted(ares, key=total) if total(x) <= FINAL * total(ab)][:NFINAL]
            whole = None
            if len(fin_f) * len(fin_a) > 1:            # the stages timed alone carry their own output copies: the finalists as whole gradients
                ft = bt = g_ref = None
                gc.collect()
                times = {}
                for f_ in fin_f:
                    for a_ in fin_a:
                        bwd = _no_backward if a_[0] == "tapes" else sweep_fns(bb[0], bb[1])[1]
                        g = population(adjoint_fn(sweep_fns(f_[0], f_[1])[0], bwd, plain, chunk, asm_fn(a_[0], a_[1])), n)
                        try:
                            e = _aot(g, tp, sz, chk, w_low, rho_u)
                            need, free = _need(e), _free()
                            if need is not None and free is not None and need > FIT * free:
                                raise RuntimeError(f"needs {need / 2**30:.2f} GiB of {free / 2**30:.2f} GiB free: not run")
                            times[(id(f_), id(a_))] = (_time(e, tp, sz, chk, w_low, rho_u, reps=5, warm=3.0), f_, a_)
                            log(f"[engine] whole gradient, forward {f_[0]} {f_[1]} + assembly {a_[0]} {a_[1]}: "
                                f"{1000 * times[(id(f_), id(a_))][0]:.1f} ms")
                        except Exception as err:
                            log(f"[engine] whole gradient, forward {f_[0]} {f_[1]} + assembly {a_[0]} {a_[1]}: unavailable "
                                f"({type(err).__name__}: {str(err).splitlines()[0][:110]})")
                        e = None
                        gc.collect()
                if times:
                    whole, fb, ab = min(times.values(), key=lambda x: x[0])
            joint = ab[0] == "tapes"
            rec = dict(engine="adjoint", fwd=[fb[0], fb[1]], bwd=(["joint", {}] if joint else [bb[0], bb[1]]), asm=[ab[0], ab[1]],
                       fwd_ms=1000 * fb[2][0], bwd_ms=0.0 if joint else 1000 * bb[2][1], assemble_ms=1000 * ab[2],
                       total_ms=1000 * (total(ab) if whole is None else whole))
        else:
            log("[engine] gradient assembly unavailable: raw JAX")
    rec["tune_s"] = time.perf_counter() - t0
    rec["compile_s"] = t_compile
    del exe, kp, ft, bt, res, timed, chk
    gc.collect()                                                    # every candidate's executable and outputs released before the next stage
    return rec


def select(tp, sz, B, w_low, rho_u, plain=None, iters=None, mode=None, log=print):
    """choose the gradient implementation for a population (tp, sz, w_low, rho_u with a leading member axis) on batches of B patients.
    Returns (grad_fn(tp, sz, pts, w_low, rho_u) -> (per-patient grads (n, B, ...), loss (n, B), J (n, B)), description).
    A shape is tuned once: another population size of the same shape re-checks (and re-times) only the choice made there. A new shape first
    tries the candidates that won everywhere so far (_preferred) and tunes the full set only when none of them works (PKPD_TUNE=full: always)."""
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
        only = None if mode == "auto" else ("xla" if mode == "adjoint" else mode)
        cands = candidates(B, n, plain, only)
        acands = asm_candidates() if mode == "auto" else [("xla", {})]
        if any(x.dtype == np.float64 for x in jax.tree_util.tree_leaves(tp)):   # the CUDA kernels compute in float32
            cands = [c for c in cands if c[0] not in ("cuda", "cblk")]
            acands = [a for a in acands if a == ("xla", {})]
        tune = lambda c, a: _tune(tp, sz, B, w_low, rho_u, plain, n, chunk, c, a, mode, log)
        prev, full = _other_size(sig), os.environ.get("PKPD_TUNE", "").lower() == "full"
        if prev is not None and not full:                                      # the same shape at another population size
            keep = lambda lst, ch: [x for x in lst if (x[0] == "xla" and not x[1]) or [x[0], x[1]] == ch]
            log(f"[engine] this shape was tuned at another population size: checking its choice")
            rec = tune(keep(cands, prev["fwd"]) + [x for x in keep(cands, prev["bwd"]) if x not in keep(cands, prev["fwd"])],
                       keep(acands, prev["asm"]))
            rec = rec if rec["engine"] == "adjoint" and [rec["fwd"], rec["asm"]] == [prev["fwd"], prev["asm"]] else None
        pref = None if full else _preferred(cands, acands)
        if rec is None and pref is not None:
            rec = tune(*pref)
            if rec["engine"] != "adjoint" or rec["fwd"][0] != "cblk" or rec["asm"][0] != "tapes":
                log("[engine] the preferred candidates do not all work at this shape: tuning every candidate")
                rec = None
        if rec is None:
            rec = tune(cands, acands)
        _save(sig, rec)
    if rec["engine"] == "raw":
        log("[engine] using raw JAX")
        return raw, rec
    a = rec.get("asm", ["xla", {}])
    log(f"[engine] using adjoint: forward {rec['fwd'][0]} {rec['fwd'][1]}, backward {rec['bwd'][0]} {rec['bwd'][1]}, assembly {a[0]} {a[1]}: "
        f"{rec['total_ms']:.0f} ms per gradient (tuning {rec['tune_s']:.0f} s)")
    bwd = _no_backward if rec["bwd"][0] == "joint" else sweep_fns(*rec["bwd"])[1]
    return population(adjoint_fn(sweep_fns(*rec["fwd"])[0], bwd, plain, chunk, asm_fn(*a)), n), rec


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
