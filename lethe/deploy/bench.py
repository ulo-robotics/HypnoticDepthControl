#!/usr/bin/env python
"""
Inference benchmark of the deployed controller: the C++ engine (cpp/lethe_infer.hpp, float and double) vs the JAX policy step, one patient,
one 5 s control step at a time (what a monitor-side controller does).

  python bench_infer.py runs/lethe/best_model.pkl [--out runs/lethe/cpp] [--repeats 2000]

Exports the model and reference episodes (deploy/cpp.py), builds cpp/bench.cpp with g++ -O3 -march=native, checks the C++ outputs against
the JAX policy step by step (max |du| in mg/min) and reports the median / p99 latency per step; then times policy.act through JAX on the CPU
and (if present) the GPU. Writes <out>/bench.json.
"""
import argparse, json, os, pickle, subprocess, time
import numpy as np

from lethe.deploy import cpp as export_cpp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository (cpp/)


def jax_latency(params, device, steps=360, reps=5):
    import jax, jax.numpy as jnp
    from lethe.core import model as M
    from lethe.core import policy as P
    with jax.default_device(device):
        ctrl = P.PolicyController(params)
        pts = M.to_numpy(M.sample_patients(jax.random.PRNGKey(0), 1, M.VAL["nominal"][0]))
        ctrl.reset(pts, None)
        ctrl.act(0, np.array([95.0]))                                 # compile
        ts = []
        for _ in range(reps):
            ctrl.reset(pts, None)
            t0 = time.perf_counter()
            for k in range(steps):
                ctrl.act(k, np.array([60.0]))
            ts.append((time.perf_counter() - t0) / steps * 1e9)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", default="")
    ap.add_argument("--repeats", type=int, default=2000)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.model)), "cpp")
    os.makedirs(out, exist_ok=True)
    from lethe.core import policy as P
    params = P.load_model(a.model)["params"]
    export_cpp.write_model(params, os.path.join(out, "model.bin"))
    export_cpp.write_trace(params, os.path.join(out, "trace.bin"))
    exe = os.path.join(out, "bench")
    subprocess.run(["g++", "-O3", "-march=native", "-std=c++17", "-o", exe, os.path.join(ROOT, "cpp", "bench.cpp")], check=True)
    res = json.loads(subprocess.run([exe, os.path.join(out, "model.bin"), os.path.join(out, "trace.bin"), str(a.repeats)],
                                    check=True, capture_output=True, text=True).stdout)
    import jax
    res["jax_cpu_ns_per_step"] = jax_latency(params, jax.devices("cpu")[0])
    try:
        res["jax_gpu_ns_per_step"] = jax_latency(params, jax.devices("gpu")[0])
    except RuntimeError:
        pass
    json.dump(res, open(os.path.join(out, "bench.json"), "w"), indent=1)
    print(f"policy H{res['H']} NS{res['NS']} L{res['L']} head{res['head']} ({res['modes']} physics modes)")
    for k in ("float", "double"):
        r = res[k]
        print(f"  C++ {k:6s}: {r['ns_per_step_median'] / 1000:8.2f} us per step (p99 {r['ns_per_step_p99'] / 1000:.2f}), "
              f"max |u - u_jax| {r['max_abs_err_vs_jax']:.1e} mg/min")
    for k in ("cpu", "gpu"):
        if f"jax_{k}_ns_per_step" in res:
            print(f"  JAX {k:6s}: {res[f'jax_{k}_ns_per_step'] / 1000:8.2f} us per step (policy.act, jitted, incl. dispatch)")
    print(f"-> {out}/bench.json")


if __name__ == "__main__":
    main()
