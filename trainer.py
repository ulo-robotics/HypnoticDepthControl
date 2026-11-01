"""
Training core shared by train.py (one configuration) and search.py (meta-optimization): configuration space, optimizers and learning-rate
rules, per-patient gradient clipping, and the Population: n configurations trained as ONE jitted program (train.py is a population of one).

Population: the members' parameters, EMAs and optimizer states are stacked along a leading axis, zero-padded to a common shape (policy.py's
training layout; masks keep every member exactly its own network). Every hyperparameter is an array with one value per member, the optimizer
is a switch over the members' update rules, K iterations run inside one lax.scan (learning-rate rule, delay curriculum and patient sampling on
the device), and all members train on the same patients each iteration (common random numbers). So one compilation serves every population of
the same shape, and many small recurrences fill the GPU together.

Configuration = optimization (opt, lr, phys_mult, ssm_mult, clip, omb1 = 1 - beta1, omb2 = 1 - beta2, wd, sched, warm, tc, floor, ema), loss
weights (w_low, rho_u), architecture (H, NS, L, head, n_extra) and prior design (q_theta, sy_scale, rho_lqr, q_x), plus the training setup
(patient_clip, nominal_frac, max_delay, delay_ramp, batch, stimulus). Gradients come from engine.select (raw JAX or the faster verified kernel paths).
"""
import json, math, os, pickle, threading, time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import optax

import model as M
import policy as P
import engine as E

jax.config.update("jax_default_matmul_precision", "highest")    # every engine computes the same float32 function (no TF32 in raw JAX)
try:
    os.makedirs(os.path.join(E.CACHE, "jax", "xla_gpu_per_fusion_autotune_cache_dir"), exist_ok=True)   # XLA does not create it on a fresh cache
    jax.config.update("jax_compilation_cache_dir", os.path.join(E.CACHE, "jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
except Exception:                                                # pragma: no cover
    pass
HOST = jax.local_devices(backend="cpu")[0]     # states are built here: a population prepared while another trains must not queue behind its GPU work

OPT_NAMES = ["adam", "adamw", "nadamw", "amsgrad", "lion", "adabelief", "radam", "lamb", "rmsprop", "sgd", "yogi"]
OPT_LR_MULT = dict(lion=0.3, rmsprop=0.3, sgd=30.0)               # lr is in Adam-equivalent units
SCHED_NAMES = ["const", "exp", "invsqrt", "cosine", "linear", "step", "wsd"]

# search space: (name, kind, a, b); log -> exp(U[log a, log b]), cat -> one of a
SPACE = [
    ("opt", "cat", OPT_NAMES, None), ("lr", "log", 1e-4, 1e-2), ("phys_mult", "log", 0.01, 1.0), ("ssm_mult", "log", 0.1, 1.0),
    ("clip", "cat", [0.25, 0.5, 1.0, 2.0, 5.0], None), ("omb1", "log", 0.01, 0.3), ("omb2", "log", 1e-4, 3e-2), ("wd", "log", 1e-4, 1e-1),
    ("sched", "cat", SCHED_NAMES, None), ("warm", "cat", [0, 25, 100], None), ("tc", "log", 100.0, 8000.0), ("floor", "log", 1e-3, 0.5),
    ("w_low", "log", 0.2, 8.0), ("rho_u", "log", 0.2, 8.0), ("ema", "cat", [0.98, 0.99, 0.995, 0.999], None),
    ("H", "cat", [8, 16, 32, 64], None), ("NS", "cat", [4, 8, 16, 32], None), ("L", "cat", [1, 2, 3], None),     # the paper's caps
    ("head", "cat", [0, 16, 64], None), ("n_extra", "cat", [0, 2, 4, 8, 16], None),
    ("q_theta", "log", 2e-4, 2e-2), ("sy_scale", "log", 0.5, 2.0), ("rho_lqr", "log", 1e-5, 1e-2), ("q_x", "log", 1e-5, 1e-3),
]
PRIOR_KEYS = ("q_theta", "sy_scale", "rho_lqr", "q_x")
SETUP = dict(patient_clip=4.0, nominal_frac=0.25, max_delay=12, delay_ramp=400, batch=512, stimulus="random")

# the selected configuration (winner of the paper's search, runs/search_full: config 171): train.py's default
SELECTED = dict(opt="nadamw", lr=7.38e-3, phys_mult=0.024, ssm_mult=0.954, clip=2.0, omb1=0.0538, omb2=8.17e-4, wd=1.55e-2, sched="wsd",
                warm=100, tc=509.1, floor=1.15e-3, ema=0.99, w_low=4.75, rho_u=0.256, H=32, NS=16, L=3, head=64, n_extra=8,
                q_theta=2.87e-3, sy_scale=0.999, rho_lqr=1.64e-3, q_x=1.40e-4, **SETUP)
# the search's starting point (trial 0 of a search; the old meta_opt DEFAULT_CFG)
SEARCH_BASE = dict(opt="adam", lr=1e-3, phys_mult=0.1, ssm_mult=0.33, clip=1.0, omb1=0.1, omb2=1e-3, wd=1e-2, sched="const", warm=0,
                   tc=1000.0, floor=0.1, ema=0.99, w_low=2.0, rho_u=2.0, H=32, NS=32, L=1, head=0, n_extra=0,
                   q_theta=2e-3, sy_scale=1.0, rho_lqr=1e-4, q_x=1e-4, **SETUP)
VAL_KEY = jax.random.PRNGKey(12345)


# ---------------------------------------------------------------- optimizer and learning-rate rules
def label(params):
    def f(path, _):
        names = [p.key for p in path if hasattr(p, "key")]
        return "phys" if names[0] == "phys" else "ssm" if names[-1] in ("log_dt", "log_A_re", "A_im") else "extra"
    return jax.tree_util.tree_map_with_path(f, params)


def _tx(name, lr, b1, b2, wd):
    return {"adam": lambda: optax.adam(lr, b1=b1, b2=b2), "adamw": lambda: optax.adamw(lr, b1=b1, b2=b2, weight_decay=wd),
            "nadamw": lambda: optax.nadamw(lr, b1=b1, b2=b2, weight_decay=wd), "amsgrad": lambda: optax.amsgrad(lr, b1=b1, b2=b2),
            "lion": lambda: optax.lion(lr, b1=b1, b2=b2, weight_decay=wd), "adabelief": lambda: optax.adabelief(lr, b1=b1, b2=b2),
            "radam": lambda: optax.radam(lr, b1=b1, b2=b2), "lamb": lambda: optax.lamb(lr, b1=b1, b2=b2, weight_decay=wd),
            "rmsprop": lambda: optax.rmsprop(lr, decay=b2, momentum=b1), "sgd": lambda: optax.sgd(lr, momentum=b1, nesterov=True),
            "yogi": lambda: optax.yogi(lr, b1=b1, b2=b2)}[name]()


def _build(name, lr_extra, lr_phys, lr_ssm, clip, b1, b2, wd):
    """three parameter groups (physics layer, SSM time scales, the rest) with their own learning rate; global-norm clip per group; weight decay
    only on the rest"""
    grp = lambda lr, w: optax.chain(optax.clip_by_global_norm(clip), _tx(name, lr, b1, b2, w))
    return optax.multi_transform({"extra": grp(lr_extra, wd), "phys": grp(lr_phys, 0.0), "ssm": grp(lr_ssm, 0.0)}, label)


# the hyperparameters live in the optimizer state (set at every step from the member's configuration)
OPTS = {n: optax.inject_hyperparams(_build, static_args="name")(name=n, lr_extra=0.0, lr_phys=0.0, lr_ssm=0.0, clip=1.0, b1=0.9, b2=0.999, wd=0.0)
        for n in OPT_NAMES}


def lr_scale(sched, tc, fl, warm, it):
    """multiplier of the base learning rate at iteration it (all traced; sched = index into SCHED_NAMES)"""
    it = it.astype(jnp.float32)
    s = jnp.stack([jnp.float32(1.0),                                                   # const
                   fl + (1 - fl) * 0.5 ** (it / tc),                                   # exp
                   jnp.maximum(fl, 1.0 / jnp.sqrt(1.0 + it / tc)),                     # invsqrt
                   fl + (1 - fl) * 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(1.0, it / tc))),   # cosine
                   jnp.maximum(fl, 1.0 - (1 - fl) * it / tc),                          # linear
                   jnp.maximum(fl, 0.5 ** jnp.floor(it / tc)),                         # step
                   1.0 - (1 - fl) * jnp.clip((it - tc) / (0.5 * tc), 0.0, 1.0)])[sched]   # wsd
    return s * jnp.where(warm > 0, jnp.minimum(1.0, (it + 1) / jnp.maximum(warm, 1.0)), 1.0)


def hyper(cfgs):
    """per-member hyperparameter arrays of a list of configurations"""
    f = lambda k: jnp.asarray([float(c[k]) for c in cfgs], jnp.float32)
    return dict(opt=jnp.asarray([OPT_NAMES.index(c["opt"]) for c in cfgs], jnp.int32),
                sched=jnp.asarray([SCHED_NAMES.index(c["sched"]) for c in cfgs], jnp.int32),
                lr=jnp.asarray([c["lr"] * OPT_LR_MULT.get(c["opt"], 1.0) for c in cfgs], jnp.float32),
                b1=1 - f("omb1"), b2=1 - f("omb2"), **{k: f(k) for k in ("phys_mult", "ssm_mult", "clip", "wd", "tc", "floor", "warm", "ema", "w_low", "rho_u")})


def opt_update(names, ost, g, params, hp, it):
    """one member: the update of its optimizer (every optimizer of the population is computed, the member's own one is kept; jnp.where, so a
    non-finite value in another rule cannot leak)"""
    base = hp["lr"] * lr_scale(hp["sched"], hp["tc"], hp["floor"], hp["warm"], it)
    hpd = dict(lr_extra=base, lr_phys=base * hp["phys_mult"], lr_ssm=base * hp["ssm_mult"], clip=hp["clip"], b1=hp["b1"], b2=hp["b2"], wd=hp["wd"])
    upd, new = None, {}
    for n in names:
        st = ost[n]._replace(hyperparams={k: jnp.asarray(v, ost[n].hyperparams[k].dtype) for k, v in hpd.items()})
        u, st2 = OPTS[n].update(g, st, params)
        if len(names) == 1:
            return {n: st2}, u
        sel = hp["opt"] == OPT_NAMES.index(n)
        new[n] = jax.tree_util.tree_map(lambda a, b: jnp.where(sel, a, b), st2, st)
        u = jax.tree_util.tree_map(lambda a: jnp.where(sel, a, 0.0), u)
        upd = u if upd is None else jax.tree_util.tree_map(jnp.add, upd, u)
    return new, upd


# ---------------------------------------------------------------- per-patient clipping
def aggregate(g, k):
    """per-patient gradients (leading axis B) -> clipped mean. Non-finite patients are dropped. Returns (gradient, clipped share)"""
    leaves, tdef = jax.tree_util.tree_flatten(g)
    B = leaves[0].shape[0]
    flat = jnp.concatenate([x.reshape(B, -1) for x in leaves], 1)
    fin = jnp.all(jnp.isfinite(flat), 1)
    flat = jnp.where(fin[:, None], flat, 0.0)
    amax = jnp.max(jnp.abs(flat), 1)
    nrm = amax * jnp.sqrt(jnp.sum((flat / jnp.where(amax > 0, amax, 1.0)[:, None]) ** 2, 1))     # no float32 overflow
    if k > 0:
        pos = fin & (nrm > 0)
        med = jnp.nanmedian(jnp.where(pos, nrm, jnp.nan))
        scale = jnp.where(pos, jnp.minimum(1.0, k * med / jnp.maximum(nrm, 1e-30)), 0.0)
        scale = jnp.where(jnp.isfinite(scale), scale, 0.0)
    else:
        scale = fin.astype(flat.dtype)
    out = jnp.sum(flat * scale[:, None], 0) / B
    parts = jnp.split(out, np.cumsum([int(np.prod(x.shape[1:])) for x in leaves])[:-1])
    return jax.tree_util.tree_unflatten(tdef, [p.reshape(x.shape[1:]) for p, x in zip(parts, leaves)]), jnp.mean(scale < 1.0)


# ---------------------------------------------------------------- states, validation
def to_np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def save(obj, path):
    with open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(path + ".tmp", path)


def new_state(c, init_key):
    """untrained member: training-layout parameters at its own sizes, EMA, its optimizer's state, iteration 0"""
    with jax.default_device(HOST):
        tp, _ = P.expand(P.init_params(init_key, c))
        return dict(params=to_np(tp), ema=to_np(tp), ost={c["opt"]: to_np(OPTS[c["opt"]].init(tp))}, it=0)


def train_spec(setup, it):
    dmax = jnp.minimum(setup["max_delay"], jnp.floor(setup["max_delay"] * it / max(setup["delay_ramp"], 1)))
    return dict(M.with_stimulus(M.TRAIN, setup["stimulus"]), dmax=dmax)


def make_validate(n, stats_fn, scenarios=None, per_patient=False):
    """validation of a population on n new patients per scenario (default: model.val_specs()), the same patients for every member.
    stats_fn(tp, sz, pts) -> per-patient (J, viol) of one member. Returns validate(tp, sz, key) -> per member {scenario: {J, viol}}"""
    scen = scenarios or M.val_specs()
    names = list(scen)
    specs = {k: jnp.asarray([scen[s][k] for s in names], jnp.float32) for k in M.SPEC_NOMINAL}

    @jax.jit
    def run(tp, sz, key):
        f = E.pop_map(stats_fn, jax.tree_util.tree_leaves(tp)[0].shape[0], (0, 0, None))

        def one(inp):
            i, sp = inp
            J, v = f(tp, sz, M.sample_patients(jax.random.fold_in(key, i), n, sp))
            return (J, v) if per_patient else (jnp.mean(J, -1), jnp.mean(v, -1))
        return jax.lax.map(one, (jnp.arange(len(names)), specs))

    exe = [run]

    def validate(tp, sz, key):
        J, v = jax.device_get(exe[0](tp, sz, key))
        conv = (lambda x: np.asarray(x)) if per_patient else float
        return [{s: dict(J=conv(J[i, m]), viol=conv(v[i, m])) for i, s in enumerate(names)} for m in range(J.shape[1])]
    validate.compile = lambda tp, sz, key: exe.__setitem__(0, run.lower(tp, sz, key).compile())       # ahead of time (optional)
    return validate


def evaluate(models, n, key, per_patient=False, stimulus="random"):
    """compact parameter trees (one network each, any sizes) on the validation scenarios, raw JAX forward; list of per-model stats"""
    val = make_validate(n, E.forward_stats_fn({}), M.val_specs(stimulus), per_patient=per_patient)
    out = []
    for p in models:
        tp, sz = P.expand(jax.tree_util.tree_map(jnp.asarray, p))
        out += val(*jax.tree_util.tree_map(lambda a: jnp.asarray(a)[None], (tp, sz)), key)
    return out


# ---------------------------------------------------------------- the population
_PROGRAMS, _PROGRAMS_LOCK = {}, threading.Lock()     # compiled training blocks and validations, shared by the populations of a process


def _program(key, build):
    """the compiled program for key, built once per process (search.py prepares populations in threads: a duplicate build is harmless)"""
    with _PROGRAMS_LOCK:
        if key in _PROGRAMS:
            return _PROGRAMS[key]
    prog = build()
    with _PROGRAMS_LOCK:
        return _PROGRAMS.setdefault(key, prog)


def own_shape(c):
    """training-layout shape of a configuration's network"""
    return dict(H=int(c["H"]), NS=int(c["NS"]), L=int(c["L"]), HD=max(int(c["head"]), 1), P=5 + int(c["n_extra"]))


def common_shape(cfgs):
    """the smallest training-layout shape that holds every configuration"""
    own = [own_shape(c) for c in cfgs]
    return {k: max(o[k] for o in own) for k in own[0]}


class Population:
    """n configurations (sharing the training setup and the data key) trained together. states: one dict(params, ema, ost, it) per member,
    training layout at any shape (new_state, or member_states of an earlier population); all members at the same iteration.
    opt_names: the optimizers whose states every member carries (default: the members' own); search.py passes all of them so that one
    compilation serves every population of a shape."""

    def __init__(self, cfgs, states, train_key, shape=None, opt_names=None, val_batch=1024, iters_hint=None, block=100, log=print):
        self.cfgs, self.n, self.log, self.block, self.key = cfgs, len(cfgs), log, block, train_key
        self.setup = {k: cfgs[0][k] for k in SETUP}
        assert all({k: c[k] for k in SETUP} == self.setup for c in cfgs), "members must share the training setup"
        its = {s["it"] for s in states}
        assert len(its) == 1, "members must be at the same iteration"
        self.it = its.pop()
        self.shape = shape or common_shape(cfgs)
        self.names = sorted(opt_names or {c["opt"] for c in cfgs}, key=OPT_NAMES.index)
        self.plain = (int(cfgs[0]["head"]) > 0) if self.n == 1 and own_shape(cfgs[0]) == self.shape else None     # one unpadded network
        with jax.default_device(HOST):                                                   # built on the host, moved to the GPU in one go
            sizes = [P.size_of(c) for c in cfgs]
            sz = {k: jnp.asarray([s[k] for s in sizes], jnp.int32) for k in sizes[0]}
            stack = lambda trees: jax.tree_util.tree_map(lambda *a: np.stack([np.asarray(x) for x in a]), *trees)
            params = [P.resize(s["params"], self.shape) for s in states]
            state = dict(params=stack(params), ema=stack([P.resize(s["ema"], self.shape) for s in states]),
                         ost={nm: stack([P.resize(s["ost"][nm], self.shape) if nm in s["ost"] else to_np(OPTS[nm].init(p))
                                         for s, p in zip(states, params)]) for nm in self.names})
            hp = hyper(cfgs)
        self.state, self.sz, self.hp, self.key = jax.device_put((state, sz, hp, train_key), jax.devices()[0])
        self.grad_fn, self.engine = E.select(self.state["params"], self.sz, self.setup["batch"], self.hp["w_low"], self.hp["rho_u"], self.plain,
                                             iters=iters_hint, log=log)
        self.stats_fn = E.forward_stats_fn(self.engine, self.plain)
        self.val_batch = val_batch
        self._validate = None
        self._blocks, self._exe = {}, {}

    def _sig(self, *extra):
        """everything a compiled block / validation depends on: two populations with the same signature run the same program (the
        hyperparameters, states and data key are arguments)"""
        spec = lambda t: jax.tree_util.tree_map(lambda a: [list(a.shape), str(a.dtype)], t)
        eng = {k: v for k, v in self.engine.items() if not k.endswith(("_ms", "_s"))}
        return json.dumps([self.n, self.shape, self.names, self.setup, eng, self.plain, spec(self.state), spec(self.sz), spec(self.hp), *extra],
                          sort_keys=True, default=str)

    def _validator(self):
        def build():
            v = make_validate(self.val_batch, self.stats_fn, M.val_specs(self.setup["stimulus"]))
            v.compile(self.state["ema"], self.sz, VAL_KEY)
            return v
        return _program(("val", self._sig(self.val_batch)), build)

    def compile(self):
        """compile the training block and the validation ahead of time (no GPU work: search.py does this for the next population while the
        current one trains); a population with the same program as an earlier one of the process reuses it"""
        K = self.block
        self._exe[K] = _program(("block", self._sig(K)),
                                lambda: self._block_fn(K).lower(self.state, self.sz, self.hp, self.key, jnp.int32(self.it)).compile())
        self._validate = self._validator()

    def _block_fn(self, K):
        """K training iterations as one jitted scan (the state is donated)"""
        if K in self._blocks:
            return self._blocks[K]
        setup, names, grad_fn, n = self.setup, self.names, self.grad_fn, self.n

        def member(g, p, e, o, hp, it):
            g, clipped = aggregate(g, setup["patient_clip"])
            o, upd = opt_update(names, o, g, p, hp, it)
            p = optax.apply_updates(p, upd)
            e = jax.tree_util.tree_map(lambda a, b: hp["ema"] * a + (1 - hp["ema"]) * b, e, p)
            return p, e, o, clipped, optax.global_norm(g)

        def it_step(carry, it, sz, hp, key):
            p, e, o = carry
            pts = M.sample_patients(jax.random.fold_in(key, it), setup["batch"], train_spec(setup, it), setup["nominal_frac"])
            g, loss, J = grad_fn(p, sz, pts, hp["w_low"], hp["rho_u"])
            p, e, o, clipped, gn = E.pop_map(member, n, (0, 0, 0, 0, 0, None))(g, p, e, o, hp, it)
            return (p, e, o), dict(loss=jnp.nanmean(loss, -1), J=jnp.nanmean(J, -1), clipped=clipped, gnorm=gn)

        def run(state, sz, hp, key, it0):
            (p, e, o), m = jax.lax.scan(lambda c, it: it_step(c, it, sz, hp, key), (state["params"], state["ema"], state["ost"]),
                                        it0 + jnp.arange(K, dtype=jnp.int32))
            return dict(params=p, ema=e, ost=o), {k: v.mean(0) for k, v in m.items()}
        self._blocks[K] = jax.jit(run, donate_argnums=(0,))
        return self._blocks[K]

    def train_to(self, n_it, on_block=None, stop=lambda: False):
        """train until iteration n_it in blocks of self.block iterations; on_block(it, per-member mean training metrics, s per iteration)"""
        while self.it < n_it and not stop():
            K = min(self.block, n_it - self.it)
            t0 = time.time()
            self.state, m = (self._exe.get(K) or self._block_fn(K))(self.state, self.sz, self.hp, self.key, jnp.int32(self.it))
            m = jax.device_get(m)
            self.it += K
            if on_block is not None:
                on_block(self.it, m, (time.time() - t0) / K)

    def validate(self, key, which="ema"):
        """per member: (stats per scenario, score)"""
        if self._validate is None:
            self._validate = self._validator()
        return [(st, M.score(st)) for st in self._validate(self.state[which], self.sz, key)]

    def member_states(self):
        """per member: dict(params, ema, ost (its own optimizer), it) at its own sizes (numpy)"""
        host = to_np(self.state)
        out = []
        for m, c in enumerate(self.cfgs):
            take = lambda t: P.resize(jax.tree_util.tree_map(lambda a: a[m], t), own_shape(c))
            out.append(dict(params=take(host["params"]), ema=take(host["ema"]), ost={c["opt"]: take(host["ost"][c["opt"]])}, it=self.it))
        return out
