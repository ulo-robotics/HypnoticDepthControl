"""
Golden outputs of a trained model, for test_golden.py (refactors must reproduce them bit for bit):

  python tests/golden.py runs/lethe/best_model.pkl --out golden/

eval.npz (per-patient metrics of the policy and the classic baselines, every validation and stress scenario, 100 patients each),
grads.npz (raw per-patient gradients, loss and J on 16 variable-length signed patients), model.bin / trace.bin (the C++ export), onnx.npz
(the ONNX graph's outputs on a fixed feed). CPU, float32 with exact matmuls.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute(model_path, out):
    import jax, jax.numpy as jnp
    jax.config.update("jax_default_matmul_precision", "highest")
    from lethe import engine as E
    from lethe.baselines import classic as BL
    from lethe.core import model as M
    from lethe.core import policy as P
    from lethe.deploy import cpp as X
    mdl = P.load_model(model_path)
    prm = jax.tree_util.tree_map(jnp.asarray, mdl["params"])
    roll = jax.jit(lambda p, pt: P.rollout(p, pt)[1:])
    ev = {}
    for si, (nm, sp) in enumerate(list((k, v[0]) for k, v in M.VAL.items()) + list(M.STRESS.items())):
        pj = M.sample_patients(jax.random.PRNGKey(4000 + si), 100, sp)
        pn = M.to_numpy(pj)
        b, u = (np.asarray(x, np.float64).T for x in roll(prm, pj))
        ev.update({f"{nm}__policy__{k}": v for k, v in M.metrics(b, u, pn["tgt"]).items()})
        for c, mk in BL.BASELINES.items():
            bb, uu = M.run_numpy(mk(), pn)
            ev.update({f"{nm}__{c}__{k}": v for k, v in M.metrics(bb, uu, pn["tgt"]).items()})
    tp, sz = P.training_form(prm)
    pts = M.sample_patients(jax.random.PRNGKey(5000), 16, dict(M.with_stimulus(M.TRAIN, "signed"), len_lo=10.0, len_hi=30.0))
    g, loss, J = jax.jit(lambda t, s, pt: E.raw_grads(t, s, pt, 4.75, 0.256))(tp, sz, pts)
    grads = dict(loss=np.asarray(loss), J=np.asarray(J),
                 **{"g/" + "/".join(str(getattr(k, "key", k)) for k in path): np.asarray(v)
                    for path, v in jax.tree_util.tree_flatten_with_path(g)[0]})
    os.makedirs(out, exist_ok=True)
    X.write_model(mdl["params"], os.path.join(out, "model.bin"))
    X.write_trace(mdl["params"], os.path.join(out, "trace.bin"), 8)
    onnx = {}
    try:
        from lethe.deploy import onnx_export as EO
        m, d = EO.build(mdl["params"])
        rng = np.random.default_rng(0)
        feed = EO._feed0(d, 64)
        feed.update(y=rng.uniform(30, 100, 64).astype(np.float32), tgt=rng.uniform(45, 60, 64).astype(np.float32),
                    tgt_prev=rng.uniform(45, 60, 64).astype(np.float32), u_prev=rng.uniform(0, 30, 64).astype(np.float32))
        for k in ("phys_re", "phys_im", "s4d_re", "s4d_im"):
            feed[k] = (0.1 * rng.standard_normal(feed[k].shape)).astype(np.float32)
        onnx = dict(**{f"in_{k}": v for k, v in feed.items()}, **{f"out_{i}": np.asarray(o) for i, o in enumerate(EO.ort_runner(m)(feed))})
    except ImportError:
        pass
    return ev, grads, onnx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev, grads, onnx = compute(a.model, a.out)
    np.savez(os.path.join(a.out, "eval.npz"), **ev)
    np.savez(os.path.join(a.out, "grads.npz"), **grads)
    if onnx:
        np.savez(os.path.join(a.out, "onnx.npz"), **onnx)
    print("golden written:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
