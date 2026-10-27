"""
Pallas (Triton) implementation of the two sweeps of kstep.py: one program per block of KB patients runs the whole 360-step episode, with the
S4D states (forward) / S4D adjoints (backward) in a per-program scratch in GPU memory and the tapes written per step. The step itself is
kstep.step / kstep.bwd_step, traced into the kernel (the backward takes jax.vjp of the replayed step inside the kernel).
Launch options (tuned by engine.py): kb patients per program, nw warps, lazy = reload every weight matrix at each use (needed at large H,
slower when everything fits).
"""
import threading
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

import model as M
import kstep as K

try:
    from jax.experimental.pallas import triton as plt
except ImportError:                                   # pragma: no cover
    plt = None


class LazyDict(dict):
    """a dict of Refs that loads a value at each access. Matrices carry a leading axis of length 1 and are loaded at an index read from memory at
    every access from its own entry of a zero table (LazyDict.zero, read at the current step): otherwise Triton merges the repeated loads of a
    matrix (forward recompute and VJP stage of the backward step; loop-invariant hoisting) into one value that stays live, and at H = 64 the weight
    matrices of all layers no longer fit into the shared memory of a block. The trace state is per thread (engine.py compiles in parallel)."""
    NZ = 64
    tl = threading.local()

    @staticmethod
    def begin_step(zero, k, lazy):
        LazyDict.tl.zero, LazyDict.tl.k, LazyDict.tl.lazy = zero, k, lazy
        LazyDict.tl.n = getattr(LazyDict.tl, "n", 0)

    def __getitem__(self, key):
        v = dict.__getitem__(self, key)
        if isinstance(v, list):
            return v
        if len(v.shape) < 3:
            return v[...]
        t = LazyDict.tl
        if not t.lazy:
            return v[0]
        t.n = (t.n + 1) % LazyDict.NZ
        return v[t.zero[t.k, t.n]]


def _whole(a):
    return pl.BlockSpec(a.shape, lambda b, n=a.ndim: (0,) * n)


def _pat(a, kb, axis):
    shp = list(a.shape)
    shp[axis] = kb
    return pl.BlockSpec(tuple(shp), lambda b, axis=axis, n=a.ndim: tuple(b if i == axis else 0 for i in range(n)))


class _Args:
    """flatten pytrees of kernel inputs / outputs with their BlockSpecs; rebuild them from the refs inside the kernel"""

    def __init__(self):
        self.arrays, self.specs, self.groups = [], [], []

    def add(self, name, tree, spec):
        leaves, tdef = jax.tree_util.tree_flatten(tree)
        self.groups.append((name, tdef, len(leaves)))
        self.arrays += leaves
        self.specs += [spec(a) for a in leaves]

    def unpack(self, refs):
        out, i = {}, 0
        for name, tdef, n in self.groups:
            out[name] = jax.tree_util.tree_unflatten(tdef, list(refs[i:i + n]))
            i += n
        return out, refs[i:]


def _kp_split(kp):
    """the scalar vector separately (loaded element by element: indexing a loaded vector would be a slice); matrices get a leading axis of 1"""
    lift = lambda t: jax.tree_util.tree_map(lambda a: a[None] if a.ndim >= 2 else a, t)
    return kp["sc"], lift({k: v for k, v in kp.items() if k != "sc"})


def _kp_in_kernel(sc_ref, kpr):
    kp = LazyDict({k: v for k, v in kpr.items() if k != "layers"})
    dict.__setitem__(kp, "sc", [sc_ref[i] for i in range(sc_ref.shape[0])])
    dict.__setitem__(kp, "layers", [LazyDict(lp) for lp in kpr["layers"]])
    return kp


def _inputs(kp, q, st, kb):
    sc, rest = _kp_split(kp)
    a = _Args()
    a.add("sc", sc, _whole)
    a.add("kp", rest, _whole)
    a.add("q", q, lambda x: _pat(x, kb, 0))
    a.add("st", st, lambda x: _pat(x, kb, 1))
    a.add("zero", jnp.zeros((M.N, LazyDict.NZ), jnp.int32), _whole)
    return a


def _params(nw):
    return plt.CompilerParams(num_warps=nw, num_stages=1) if plt is not None else None


def forward(kp, q, st, tape=True, kb=16, nw=4, lazy=False, interpret=False):
    """same contract as kstep.xla_forward"""
    B = q["mu"].shape[0]
    H, NS = kp["layers"][0]["Ar"].shape
    L, Pp, dt = len(kp["layers"]), kp["lamr"].shape[0], q["mu"].dtype
    ins = _inputs(kp, q, st, kb)
    c0 = K.carry0(B, Pp, dt)
    outs = _Args()
    if tape:
        outs.add("c", {k: jax.ShapeDtypeStruct((M.N,) + v.shape, dt) for k, v in c0.items()}, lambda x: _pat(x, kb, 1))
        outs.add("ys", [jax.ShapeDtypeStruct((M.N, B, H), dt)] * L, lambda x: _pat(x, kb, 1))
    outs.add("sums", [jax.ShapeDtypeStruct((B,), dt)] * 3, lambda x: _pat(x, kb, 0))
    outs.add("h", [jax.ShapeDtypeStruct((B, H, NS), dt)] * (2 * L), lambda x: _pat(x, kb, 0))      # scratch

    def kernel(*refs):
        d, orefs = ins.unpack(refs)
        o, _ = outs.unpack(orefs)
        kpk = _kp_in_kernel(d["sc"], d["kp"])
        qv = {k: v[...] for k, v in d["q"].items()}
        hr = o["h"]
        for r in hr:
            r[...] = jnp.zeros(r.shape, dt)

        def body(k, carry):
            c, s_cost, s_se, s_v = carry
            LazyDict.begin_step(d["zero"], k, lazy)
            s = {n: v[k] for n, v in d["st"].items()}
            if tape:
                for n, v in c.items():
                    o["c"][n][k] = v

            def s4d(l, u):
                ys, (h2r, h2i) = K.s4d_update(kpk["layers"][l], hr[2 * l][...], hr[2 * l + 1][...], u)
                hr[2 * l][...] = h2r
                hr[2 * l + 1][...] = h2i
                if tape:
                    o["ys"][l][k] = ys
                return ys, None
            c2, cost, b, _, _ = K.step(kpk, qv, c, k, s, s4d)
            return c2, s_cost + cost, s_se + (b - s["tg"]) ** 2, s_v + jax.nn.relu(40 - b) ** 2

        z = jnp.zeros((kb,), dt)
        _, s_cost, s_se, s_v = lax.fori_loop(0, M.N, body, (K.carry0(kb, Pp, dt), z, z, z))
        for r, v in zip(o["sums"], (s_cost, s_se, s_v)):
            r[...] = v

    res = pl.pallas_call(kernel, grid=(B // kb,), in_specs=ins.specs, out_specs=outs.specs,
                         out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype) for x in outs.arrays],
                         compiler_params=_params(nw), interpret=interpret, name="pkpd_fwd")(*ins.arrays)
    r, _ = outs.unpack(res)
    sums = dict(cost=r["sums"][0], se=r["sums"][1], viol=r["sums"][2])
    return (dict(c=r["c"], ys=r["ys"]) if tape else None), sums


def backward(kp, q, st, ftape, kb=16, nw=4, lazy=False, interpret=False):
    """same contract as kstep.xla_backward"""
    B = q["mu"].shape[0]
    H, NS = kp["layers"][0]["Ar"].shape
    L, Pp, dt = len(kp["layers"]), kp["lamr"].shape[0], q["mu"].dtype
    ins = _inputs(kp, q, st, kb)
    ins.add("c", ftape["c"], lambda x: _pat(x, kb, 1))
    ins.add("ys", ftape["ys"], lambda x: _pat(x, kb, 1))
    c0 = K.carry0(B, Pp, dt)
    outs = _Args()
    outs.add("lam", {k: jax.ShapeDtypeStruct((M.N,) + v.shape, dt) for k, v in c0.items()}, lambda x: _pat(x, kb, 1))
    outs.add("e", [jax.ShapeDtypeStruct((M.N, B, H), dt)] * L, lambda x: _pat(x, kb, 1))
    outs.add("h", [jax.ShapeDtypeStruct((B, H, NS), dt)] * (2 * L), lambda x: _pat(x, kb, 0))      # scratch: lam_h

    def kernel(*refs):
        d, orefs = ins.unpack(refs)
        o, _ = outs.unpack(orefs)
        kpk = _kp_in_kernel(d["sc"], d["kp"])
        qv = {k: v[...] for k, v in d["q"].items()}
        hr = o["h"]
        for r in hr:
            r[...] = jnp.zeros(r.shape, dt)

        def body(i, lam_c):
            k = M.N - 1 - i
            LazyDict.begin_step(d["zero"], k, lazy)
            s = {n: v[k] for n, v in d["st"].items()}
            c = {n: v[k] for n, v in d["c"].items()}
            ys = [v[k] for v in d["ys"]]
            for n, v in lam_c.items():
                o["lam"][n][k] = v
            lam_h = [(hr[2 * l][...], hr[2 * l + 1][...]) for l in range(L)]
            e, c_bar, new_h = K.bwd_step(kpk, qv, c, k, s, ys, lam_c, lam_h, 1.0 / M.N)
            for l in range(L):
                o["e"][l][k] = e[l]
                hr[2 * l][...] = new_h[l][0]
                hr[2 * l + 1][...] = new_h[l][1]
            return c_bar

        lam0 = jax.tree_util.tree_map(jnp.zeros_like, K.carry0(kb, Pp, dt))
        lax.fori_loop(0, M.N, body, lam0)

    res = pl.pallas_call(kernel, grid=(B // kb,), in_specs=ins.specs, out_specs=outs.specs,
                         out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype) for x in outs.arrays],
                         compiler_params=_params(nw), interpret=interpret, name="pkpd_bwd")(*ins.arrays)
    r, _ = outs.unpack(res)
    return dict(lam=r["lam"], e=r["e"])
