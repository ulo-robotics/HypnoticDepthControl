"""
Lethe on a Coral USB Edge TPU (board side; numpy and tflite_runtime only; edge.py run --what coral): the recurrences, GELUs, residual
connections, layer norms and the decoder on the CPU in float32; the int8 gated mixing of every layer and the head's first layer
(lethe.deploy.edge.coral, compiled by edgetpu_compiler) on the TPU: L + 1 calls per step.

  python3 bench.py <dir> [repeats]
<dir>: arrays.npz + dims.json (the discretized model), trace.bin (the JAX policy's recorded closed loops), glu{l}.tflite, head.tflite
(int8, CPU) and their *_edgetpu.tflite (compiled); libedgetpu.so.1 from the library path or EDGETPU_LIB. Replays the recorded
measurements and targets (states fed back) with the TPU parts on the CPU and on the TPU, the TPU's int8 outputs compared on every call with
the CPU's on the same inputs, reports the largest difference from the JAX trace, then times `repeats` episodes: per step (CPU part
included) and per TPU call. Prints one JSON line per engine (cpu_int8, edgetpu).
"""
import json, os, sys, time
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:                                           # the same interpreter from the LiteRT package
    from ai_edge_litert.interpreter import Interpreter, load_delegate

E0 = EMAX = 100.0
GAMMA, C50N, UMAX = 2.0, 4.5, 100.0
C = 0.7978845608028654


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(C * (x + 0.044715 * x * x * x)))


def wref(t):
    s = (E0 - t) / EMAX
    return C50N * (s / (1 - s)) ** (1 / GAMMA)


class Block:
    """an int8 model of one input as a float function; returns (float output, int8 output)"""

    def __init__(self, path, delegate=None):
        self.it = Interpreter(model_path=path, experimental_delegates=[delegate] if delegate else None)
        self.it.allocate_tensors()
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.ns, self.check = [], False

    def __call__(self, v):
        s, z = self.i["quantization"]
        self.it.set_tensor(self.i["index"], np.clip(np.round(v / s + z), -128, 127).astype(np.int8)[None])
        t0 = time.perf_counter_ns()
        self.it.invoke()
        self.ns.append(time.perf_counter_ns() - t0)
        s, z = self.o["quantization"]
        q = self.it.get_tensor(self.o["index"])[0]
        return (q.astype(np.float32) - z) * s, q


class Checked:
    """the TPU block, and the CPU block on the same quantized inputs: the largest difference of their int8 outputs (LSB)"""

    def __init__(self, tpu, cpu):
        self.tpu, self.cpu, self.lsb, self.check = tpu, cpu, 0, True
        self.ns = tpu.ns

    def __call__(self, v):
        out, q = self.tpu(v)
        if self.check:
            self.lsb = max(self.lsb, int(np.abs(q.astype(np.int32) - self.cpu(v)[1].astype(np.int32)).max()))
        return out, q


class Policy:
    """one patient, float32 numpy (cpp/lethe_infer.hpp's step) with the blocks plugged in"""

    def __init__(self, a, d, blocks):
        self.a, self.d, self.blocks = a, d, blocks
        self.reset()

    def reset(self):
        d = self.d
        self.pr = np.zeros(d["modes"], np.float32)
        self.pi = np.zeros(d["modes"], np.float32)
        self.sr = np.zeros((d["L"], d["H"], d["NS"]), np.float32)
        self.si = np.zeros((d["L"], d["H"], d["NS"]), np.float32)
        self.up = np.float32(0)

    def step(self, y, tgt, tp):
        a = self.a
        r_, rho = wref(tgt) / C50N, wref(tp) / wref(tgt)
        self.pr *= rho
        self.pi *= rho
        s = min(max((E0 - y) / EMAX, 1e-3), 1 - 1e-3)
        w = min(C50N * (s / (1 - s)) ** (1 / GAMMA), 3 * C50N)
        x = np.float32([(y - tgt) / 50, self.up / (r_ * UMAX), w / (r_ * C50N) - 1]) @ a["enc_W"] + a["enc_b"]
        for l in range(self.d["L"]):
            Ar, Ai, Br, Bi, Cr, Ci = a[f"A{l}"]
            nr = Ar * self.sr[l] - Ai * self.si[l] + Br * x[:, None]
            ni = Ar * self.si[l] + Ai * self.sr[l] + Bi * x[:, None]
            self.sr[l], self.si[l] = nr, ni
            rr = 2 * (nr * Cr - ni * Ci).sum(1) + a[f"l{l}_D"] * x
            x = x + self.blocks[l](gelu(rr).astype(np.float32))[0]
            x = x - x.mean()
            x = x / np.sqrt((x * x).mean() + 1e-5) * a[f"l{l}_ln_g"] + a[f"l{l}_ln_b"]
        o = float(gelu(self.blocks[-1](x.astype(np.float32))[0].reshape(-1)) @ a["dec_W"] + a["dec_b"])
        ul = float(a["u0"] + (self.pr * a["Cout_re"] - self.pi * a["Cout_im"]).sum())
        u = min(max(r_ * (ul + 10 * o), 0.0), UMAX)
        un, wn = u / r_, w / r_
        lr, li = a["lam_re"], a["lam_im"]
        self.pr, self.pi = (lr * self.pr - li * self.pi + a["Bu_re"] * un + a["Bw_re"] * wn,
                            lr * self.pi + li * self.pr + a["Bu_im"] * un + a["Bw_im"] * wn)
        self.up = np.float32(u)
        return u


def main():
    d_, reps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
    d = json.load(open(os.path.join(d_, "dims.json")))
    a = {k: v.astype(np.float32) for k, v in np.load(os.path.join(d_, "arrays.npz")).items()}
    for l in range(d["L"]):
        a[f"A{l}"] = tuple(a[f"l{l}_{k}"] for k in ("dA_re", "dA_im", "dB_re", "dB_im", "C_re", "C_im"))
    raw = open(os.path.join(d_, "trace.bin"), "rb").read()
    n, N = np.frombuffer(raw[:8], "<i4")
    tr = np.frombuffer(raw[8:], "<f8").reshape(n, 3, N)
    lib = os.environ.get("EDGETPU_LIB", "libedgetpu.so.1")
    names = [f"glu{l}" for l in range(d["L"])] + ["head"]
    engines = {"cpu_int8": lambda nm: Block(os.path.join(d_, f"{nm}.tflite"))}
    if os.path.exists(os.path.join(d_, "head_edgetpu.tflite")):
        dl = load_delegate(lib)
        engines["edgetpu"] = lambda nm: Checked(Block(os.path.join(d_, f"{nm}_edgetpu.tflite"), dl),
                                                Block(os.path.join(d_, f"{nm}.tflite")))
    res = {}
    for name, make in engines.items():
        blocks = [make(nm) for nm in names]
        pol = Policy(a, d, blocks)
        err = 0.0
        for i in range(n):                                    # correctness: every recorded patient once
            pol.reset()
            for k in range(N):
                u = pol.step(tr[i, 0, k], tr[i, 1, k], tr[i, 1, max(k - 1, 0)])
                err = max(err, abs(u - tr[i, 2, k]))
        for b in blocks:
            b.ns.clear()
            b.check = False
        steps = []
        for r in range(reps):                                 # latency
            i = r % n
            pol.reset()
            for k in range(N):
                t0 = time.perf_counter_ns()
                pol.step(tr[i, 0, k], tr[i, 1, k], tr[i, 1, max(k - 1, 0)])
                steps.append(time.perf_counter_ns() - t0)
        st, inv = np.array(steps) / 1e3, np.concatenate([b.ns for b in blocks]) / 1e3
        res[name] = dict(engine=name, max_abs_err_vs_jax=float(err), us_per_step_mean=float(st.mean()),
                         us_per_step_median=float(np.median(st)), us_per_step_p99=float(np.percentile(st, 99)),
                         us_per_step_max=float(st.max()), us_per_call_median=float(np.median(inv)),
                         us_per_call_p99=float(np.percentile(inv, 99)), calls_per_step=len(names), steps=len(steps))
        if name == "edgetpu":
            res[name]["max_lsb_vs_cpu_int8"] = max(b.lsb for b in blocks)
    for r in res.values():
        print(json.dumps(r))


if __name__ == "__main__":
    main()
