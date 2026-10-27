#!/usr/bin/env python
"""
Export a trained policy for the C++ inference engine (cpp/lethe_infer.hpp), and record reference episodes to check it against.

  python export_cpp.py runs/lethe/best_model.pkl --out runs/lethe/cpp        # -> model.bin, trace.bin

model.bin: "LETHE01\\0", int32 H, NS, L, head, modes, then float64 arrays (the discretized matrices, computed here in float64):
  u0, dec_b | lam re / im, Bu re / im, Bw re / im, Cout re / im [modes] | enc_W (4, H), enc_b | per layer: dA re / im, dB re / im,
  C re / im (H, NS), D, W_mix (H, 2H), b_mix, ln_g, ln_b | (head) head_W1 (H, head), head_b1 | dec_W
trace.bin: int32 n_patients, n_steps, then per patient float64 measured BIS[n], target[n], infusion[n] of the JAX policy (policy.act) in a
closed loop on the numpy reference plant (model.Plant), patients from the mixed_hard scenario.
"""
import argparse, os, pickle, struct
import numpy as np


def write_model(params, path):
    ph, ex = params["phys"], params["extra"]
    f64 = lambda a: np.asarray(a, np.float64)
    lam = np.exp(-np.exp(f64(ph["a"])) + 1j * f64(ph["b"]))
    cx = lambda m: f64(m)[:, 0] + 1j * f64(m)[:, 1]
    H, NS = f64(ex["layers"][0]["log_A_re"]).shape
    hd = f64(ex["head_b1"]).shape[0] if "head_W1" in ex else 0
    parts = [f64([ph["u0"][0], ex["dec_b"][0]])]
    for z in (lam, cx(ph["Bu"]), cx(ph["Bw"]), cx(ph["Cout"])):
        parts += [z.real, z.imag]
    parts += [f64(ex["enc_W"]), f64(ex["enc_b"])]
    for lp in ex["layers"]:
        A = -np.exp(f64(lp["log_A_re"])) + 1j * f64(lp["A_im"])
        dA = np.exp(np.exp(f64(lp["log_dt"]))[:, None] * A)
        dB = (dA - 1) / A
        C = cx(lp["C"].reshape(-1, 2)).reshape(H, NS)
        parts += [dA.real, dA.imag, dB.real, dB.imag, C.real, C.imag, f64(lp["D"]), f64(lp["W_mix"]), f64(lp["b_mix"]), f64(lp["ln_g"]), f64(lp["ln_b"])]
    if hd:
        parts += [f64(ex["head_W1"]), f64(ex["head_b1"])]
    parts.append(f64(ex["dec_W"])[:, 0])
    with open(path, "wb") as f:
        f.write(b"LETHE01\0" + struct.pack("<5i", H, NS, len(ex["layers"]), hd, len(lam)))
        for p in parts:
            f.write(np.ascontiguousarray(p, "<f8").tobytes())


def write_trace(params, path, n=8, seed=0):
    import jax
    import model as M, policy as P
    jax.config.update("jax_default_matmul_precision", "highest")      # the reference in exact float32 (GPUs default to TF32 matmuls)
    pts = M.to_numpy(M.sample_patients(jax.random.PRNGKey(seed), n, M.VAL["mixed_hard"][0]))
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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    params = pickle.load(open(a.model, "rb"))["params"]
    write_model(params, os.path.join(a.out, "model.bin"))
    write_trace(params, os.path.join(a.out, "trace.bin"), a.patients)
    print(f"wrote {a.out}/model.bin and trace.bin")


if __name__ == "__main__":
    main()
