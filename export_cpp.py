#!/usr/bin/env python
"""
Export a trained policy for the C++ inference engine (cpp/lethe_infer.hpp), and record reference episodes to check it against.

  python export_cpp.py runs/lethe/best_model.pkl --out runs/lethe/cpp        # -> model.bin, trace.bin
  --semantics legacy|current: what the model computes beyond its parameters (model.SEMANTICS; required unless the .pkl records it)

model.bin: "LETHE02\\0", int32 H, NS, L, head, modes, flags (bit 0: the time feature), reserved, then float64 arrays (the discretized matrices, computed here in float64):
  u0, dec_b | lam re / im, Bu re / im, Bw re / im, Cout re / im [modes] | enc_W (4, H), enc_b | per layer: dA re / im, dB re / im,
  C re / im (H, NS), D, W_mix (H, 2H), b_mix, ln_g, ln_b | (head) head_W1 (H, head), head_b1 | dec_W
trace.bin: int32 n_patients, n_steps, then per patient float64 measured BIS[n], target[n], infusion[n] of the JAX policy (policy.act) in a
closed loop on the numpy reference plant (model.Plant), patients from the mixed_hard scenario.
"""
import argparse, os, pickle, struct
import numpy as np


def discretized(params, sem):
    """the deployed model in float64: sizes (H, NS, L, head, modes) and the arrays of model.bin in their order, by name (the discretized
    matrices under the semantics sem, model.SEMANTICS: S4D time constants bounded by tau_max > 0; shared with export_onnx.py)"""
    ph, ex = params["phys"], params["extra"]
    f64 = lambda a: np.asarray(a, np.float64)
    lam = np.exp(-np.exp(f64(ph["a"])) + 1j * f64(ph["b"]))
    cx = lambda m: f64(m)[:, 0] + 1j * f64(m)[:, 1]
    H, NS = f64(ex["layers"][0]["log_A_re"]).shape
    hd = f64(ex["head_b1"]).shape[0] if "head_W1" in ex else 0
    a = dict(u0=f64(ph["u0"][0]), dec_b=f64(ex["dec_b"][0]))
    for n, z in (("lam", lam), ("Bu", cx(ph["Bu"])), ("Bw", cx(ph["Bw"])), ("Cout", cx(ph["Cout"]))):
        a[n + "_re"], a[n + "_im"] = z.real, z.imag
    a["enc_W"], a["enc_b"] = f64(ex["enc_W"]), f64(ex["enc_b"])
    for l, lp in enumerate(ex["layers"]):
        dt = np.exp(f64(lp["log_dt"]))[:, None]
        if sem["tau_max"] > 0:                                          # the bounded decay per step, as model.s4d_A
            rate = np.maximum(dt * np.exp(f64(lp["log_A_re"])), (5.0 / 60.0) / sem["tau_max"])
            A = -rate / dt + 1j * f64(lp["A_im"])
        else:
            A = -np.exp(f64(lp["log_A_re"])) + 1j * f64(lp["A_im"])
        dA = np.exp(dt * A)
        dB = (dA - 1) / A
        C = cx(lp["C"].reshape(-1, 2)).reshape(H, NS)
        for n, v in (("dA_re", dA.real), ("dA_im", dA.imag), ("dB_re", dB.real), ("dB_im", dB.imag), ("C_re", C.real), ("C_im", C.imag),
                     ("D", lp["D"]), ("W_mix", lp["W_mix"]), ("b_mix", lp["b_mix"]), ("ln_g", lp["ln_g"]), ("ln_b", lp["ln_b"])):
            a[f"l{l}_{n}"] = f64(v)
    if hd:
        a["head_W1"], a["head_b1"] = f64(ex["head_W1"]), f64(ex["head_b1"])
    a["dec_W"] = f64(ex["dec_W"])[:, 0]
    return dict(H=H, NS=NS, L=len(ex["layers"]), head=hd, modes=len(lam)), a


def semantics(model, name=None):
    """a saved model's semantics (model.semantics_of)"""
    import model as M
    return M.semantics_of(model, name)


def write_model(params, path, sem):
    d, a = discretized(params, sem)
    with open(path, "wb") as f:
        f.write(b"LETHE02\0" + struct.pack("<7i", d["H"], d["NS"], d["L"], d["head"], d["modes"], int(bool(sem["time_feature"])), 0))
        f.write(np.ascontiguousarray([a["u0"], a["dec_b"]], "<f8").tobytes())
        for k in [k for k in a if k not in ("u0", "dec_b")]:
            f.write(np.ascontiguousarray(a[k], "<f8").tobytes())


def write_trace(params, path, sem, n=8, seed=0):
    import jax
    import model as M, policy as P
    jax.config.update("jax_default_matmul_precision", "highest")      # the reference in exact float32 (GPUs default to TF32 matmuls)
    pts = M.to_numpy(M.sample_patients(jax.random.PRNGKey(seed), n, M.VAL["mixed_hard"][0]))
    P.use_semantics(sem)
    ctrl = P.PolicyController(params)
    plant = M.Plant(pts)
    ctrl.reset(pts, plant)
    ys, us = np.zeros((n, M.N)), np.zeros((n, M.N))
    for k in range(M.N):
        y = plant.measure()
        u = np.clip(ctrl.act(k, y), 0.0, M.UMAX)
        ys[:, k], us[:, k] = y, u
        plant.step(u)
    with open(path, "wb") as f:
        f.write(struct.pack("<2i", n, M.N))
        for i in range(n):
            for a in (ys[i], pts["tgt"][i], us[i]):
                f.write(np.ascontiguousarray(a, "<f8").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=8)
    ap.add_argument("--semantics", choices=["legacy", "current"], default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = pickle.load(open(a.model, "rb"))
    sem = semantics(m, a.semantics)
    write_model(m["params"], os.path.join(a.out, "model.bin"), sem)
    write_trace(m["params"], os.path.join(a.out, "trace.bin"), sem, a.patients)
    print(f"wrote {a.out}/model.bin and trace.bin")


if __name__ == "__main__":
    main()
