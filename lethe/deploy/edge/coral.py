"""
The policy on a Coral Edge TPU: the network's dense layers in int8 on the TPU, everything else in float32 on the host CPU.

Per step, CPU: the features, the encoder, every S4D recurrence and its readout r = 2 Re(C s) + D x, the GELUs, the residual connections
and layer norms, the decoder, the physics modes and the infusion. TPU (int8, batch 1): glu{l} maps a = GELU(r) to the gated mixing
(W_v a + b_v) * sigmoid(W_g a + b_g), one call per layer; head maps the last layer's output x to W_1 x + b_1. L + 1 calls per step.

Only matrix products, additions, products of two tensors and the logistic run on the TPU: with every constant a full-length vector, these
give the same int8 results as TFLite's CPU kernels, bit for bit. Not on the TPU: element-wise chains like GELU (the TPU rounds the
intermediate rescalings differently and tanh's slope amplifies it), broadcast scalar constants (mishandled by edgetpu_compiler 14.1, the
version available) and int8 RSQRT (layer norm; not supported by 14.1). The recurrent states stay in float: in int8 they break the
controller (lethe.deploy.onnx_export).

build() writes, into one directory (what the board needs, edge/coral/bench.py):
  glu{l}.tflite, head.tflite                  int8 models (calibrated on closed loops of patients of their own keys)
  glu{l}_edgetpu.tflite, head_edgetpu.tflite  the same compiled by edgetpu_compiler (x86-64 only; github.com/google-coral/edgetpu,
                                              compiler/x86_64), with its logs; skipped when no compiler is given or found
  arrays.npz, dims.json                       the discretized model (lethe.deploy.cpp.discretized)
and checks the split in closed loops on the numpy plant over every validation scenario: the float split against onnxruntime float32, and the
int8 models (TFLite CPU kernels, which the TPU reproduces) against the float split, next to the floor of a BIS read as an integer.
Needs TensorFlow (the converter and the reference interpreter) and onnxruntime.
"""
import json, os, re, shutil, subprocess
import numpy as np

from lethe.core import model as M

E0, EMAX, GAMMA, C50N, UMAX = M.E0, M.EMAX, M.GAMMA, M.C50N, M.UMAX
C = 0.7978845608028654                                          # sqrt(2 / pi): GELU's tanh approximation, as jax.nn.gelu


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(C * (x + 0.044715 * x * x * x)))


def _wref(t):
    return C50N * (((E0 - t) / EMAX) / (1 - (E0 - t) / EMAX)) ** (1 / GAMMA)


class Split:
    """the step with the ONNX interface (lethe.deploy.onnx_export.loop), batched, float32 numpy; the TPU parts (glus, head) are
    pluggable float functions; tap(name, input) sees every input of a TPU part"""

    def __init__(self, a, d, glus=None, head=None, tap=None):
        self.a, self.d, self.tap = {k: np.asarray(v, np.float32) for k, v in a.items()}, d, tap
        H = d["H"]
        self.glus = glus or [self._glu(self.a[f"l{l}_W_mix"].reshape(H, 2 * H), self.a[f"l{l}_b_mix"]) for l in range(d["L"])]
        self.head = head or (lambda x: x @ self.a["head_W1"].reshape(H, -1) + self.a["head_b1"])

    def _glu(self, W, b):
        H = self.d["H"]

        def fn(v):
            z = v @ W + b
            return z[:, :H] / (1 + np.exp(-z[:, H:]))
        return fn

    def __call__(self, feed):
        a, d = self.a, self.d
        y, tgt, tp, up = feed["y"], feed["tgt"], feed["tgt_prev"], feed["u_prev"]
        r_, rho = _wref(tgt) / C50N, _wref(tp) / _wref(tgt)
        pr, pi = feed["phys_re"] * rho[:, None], feed["phys_im"] * rho[:, None]
        s = np.clip((E0 - y) / EMAX, 1e-3, 1 - 1e-3)
        w = np.minimum(C50N * (s / (1 - s)) ** (1 / GAMMA), 3 * C50N)
        x = np.stack([(y - tgt) / 50, up / (r_ * UMAX), w / (r_ * C50N) - 1], 1) @ a["enc_W"] + a["enc_b"]
        sr, si = feed["s4d_re"].copy(), feed["s4d_im"].copy()
        for l in range(d["L"]):
            Ar, Ai, Br, Bi = (a[f"l{l}_{k}"] for k in ("dA_re", "dA_im", "dB_re", "dB_im"))
            nr = Ar * sr[l] - Ai * si[l] + Br * x[:, :, None]
            ni = Ar * si[l] + Ai * sr[l] + Bi * x[:, :, None]
            sr[l], si[l] = nr, ni
            g = gelu(2 * (nr * a[f"l{l}_C_re"] - ni * a[f"l{l}_C_im"]).sum(2) + a[f"l{l}_D"] * x).astype(np.float32)
            if self.tap:
                self.tap(f"glu{l}", g)
            x = x + self.glus[l](g)
            x = x - x.mean(1, keepdims=True)
            x = x / np.sqrt((x * x).mean(1, keepdims=True) + 1e-5) * a[f"l{l}_ln_g"] + a[f"l{l}_ln_b"]
        x = x.astype(np.float32)
        if self.tap:
            self.tap("head", x)
        o = gelu(self.head(x)) @ a["dec_W"] + a["dec_b"]
        ul = a["u0"] + (pr * a["Cout_re"] - pi * a["Cout_im"]).sum(1)
        u = np.clip(r_ * (ul + 10 * o), 0, UMAX)
        un, wn = u / r_, w / r_
        lr, li = a["lam_re"], a["lam_im"]
        npr = lr * pr - li * pi + a["Bu_re"] * un[:, None] + a["Bw_re"] * wn[:, None]
        npi = lr * pi + li * pr + a["Bu_im"] * un[:, None] + a["Bw_im"] * wn[:, None]
        return [u.astype(np.float32), npr.astype(np.float32), npi.astype(np.float32), sr.astype(np.float32), si.astype(np.float32)]


def _glu_tf(tf, Wv, Wg, bv, bg):
    def glu(v):
        return (tf.matmul(v, Wv) + bv) * tf.sigmoid(tf.matmul(v, Wg) + bg)
    return glu


def _head_tf(tf, W1, b1):
    def head(x):
        return tf.matmul(x, W1) + b1
    return head


def tf_models(a, d):
    """name -> tf.function of the TPU parts (batch 1; every constant a full-length row, never a broadcast scalar)"""
    import tensorflow as tf
    H = d["H"]
    c = lambda v: tf.constant(np.asarray(v, np.float32))
    spec = lambda nm: [tf.TensorSpec([1, H], tf.float32, name=nm)]
    out = {}
    for l in range(d["L"]):
        W, b = np.asarray(a[f"l{l}_W_mix"], np.float32).reshape(H, 2 * H), np.asarray(a[f"l{l}_b_mix"], np.float32)
        out[f"glu{l}"] = tf.function(_glu_tf(tf, c(W[:, :H]), c(W[:, H:]), c(b[None, :H]), c(b[None, H:])), input_signature=spec("a"))
    W1, b1 = c(np.asarray(a["head_W1"]).reshape(H, -1)), c(np.reshape(a["head_b1"], (1, -1)))
    out["head"] = tf.function(_head_tf(tf, W1, b1), input_signature=spec("x"))
    return out


def to_int8(fn, samples):
    """full-integer int8 TFLite model (int8 input and output), activations calibrated on `samples` (rows)"""
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_concrete_functions([fn.get_concrete_function()], fn)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([v[None]] for v in samples)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    return conv.convert()


class TFLiteModel:
    """an int8 model as a float function of a batch (rows one at a time: the models are batch 1), on TFLite's CPU kernels"""

    def __init__(self, path):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.i, self.o = self.it.get_input_details()[0], self.it.get_output_details()[0]

    def __call__(self, v):
        si, zi = self.i["quantization"]
        so, zo = self.o["quantization"]
        res = []
        for k in range(v.shape[0]):
            self.it.set_tensor(self.i["index"], np.clip(np.round(v[k:k + 1] / si + zi), -128, 127).astype(np.int8))
            self.it.invoke()
            res.append((self.it.get_tensor(self.o["index"]).astype(np.float32) - zo) * so)
        return np.concatenate(res, 0)


def compile_edgetpu(path, compiler):
    """edgetpu_compiler on one model -> {op: 'Mapped to Edge TPU' | other status} (raises if it fails)"""
    r = subprocess.run([compiler, "-s", "-o", os.path.dirname(path) or ".", path], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"edgetpu_compiler failed on {path}: {r.stdout}{r.stderr}")
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^(\w+)\s+\d+\s+(.+)$", r.stdout, re.M)}


def build(params, out, patients=32, compiler=None, log=print):
    """the int8 models (compiled when an edgetpu_compiler is given or on the PATH), the arrays for the board, and the closed-loop checks;
    -> dict of the checks"""
    from lethe.deploy import cpp, onnx_export as OX
    os.makedirs(out, exist_ok=True)
    d, a = cpp.discretized(params)
    if not d["head"]:
        raise SystemExit("coral: the split assumes an MLP head (head > 0)")
    np.savez(os.path.join(out, "arrays.npz"), **{k: np.asarray(v) for k, v in a.items()})
    json.dump({k: int(d[k]) for k in ("H", "NS", "L", "head", "modes")}, open(os.path.join(out, "dims.json"), "w"))
    sc = OX.scenarios(patients, seed=0)
    ref = Split(a, d)
    lf = {s: OX.loop(ref, pts, d) for s, pts in sc.items()}
    r32 = OX.ort_runner(OX.build(params)[0])
    res = {"float_vs_onnxruntime": OX.compare(lf, {s: OX.loop(r32, pts, d) for s, pts in sc.items()}, sc)}
    log("coral: float split vs onnxruntime float32: dscore %+.1e  |du| p99 %.1e max %.1e  |dBIS| p99 %.3f max %.3f" % res["float_vs_onnxruntime"])
    cal = {}                                                        # calibration: the float split's closed loops, patients of their own keys
    tap = Split(a, d, tap=lambda nm, v: cal.setdefault(nm, []).extend(v[::7]))
    for pts in OX.scenarios(patients, seed=1000).values():
        OX.loop(tap, pts, d)
    rng, paths = np.random.default_rng(0), {}
    for nm, fn in tf_models(a, d).items():
        rows = [np.float32(cal[nm][i]) for i in rng.choice(len(cal[nm]), min(2000, len(cal[nm])), replace=False)]
        paths[nm] = os.path.join(out, f"{nm}.tflite")
        open(paths[nm], "wb").write(to_int8(fn, rows))
    q = Split(a, d, glus=[TFLiteModel(paths[f"glu{l}"]) for l in range(d["L"])], head=TFLiteModel(paths["head"]))
    res["floor_integer_bis"] = OX.compare({s: OX.loop(ref, pts, d, y_int=True) for s, pts in sc.items()}, lf, sc)
    res["int8_vs_float"] = OX.compare({s: OX.loop(q, pts, d) for s, pts in sc.items()}, lf, sc)
    log(f"coral: closed loops, {patients} patients x {len(sc)} validation scenarios, against the float split (score {OX.clinical(lf, sc):.4f}):")
    for k, name in (("floor_integer_bis", "floor: BIS read as an integer"), ("int8_vs_float", "int8 TPU parts (TFLite CPU kernels)")):
        log(f"  {name:38s} dscore %+.2e  |du| p99 %.2f max %.2f  |dBIS| p99 %.3f max %.2f" % res[k])
    compiler = compiler or shutil.which("edgetpu_compiler")
    if compiler:
        res["mapping"] = {nm: compile_edgetpu(p, compiler) for nm, p in paths.items()}
        off = {nm: {op: st for op, st in m.items() if st != "Mapped to Edge TPU"} for nm, m in res["mapping"].items()}
        log("coral: compiled " + ", ".join(paths) + ("; every op on the Edge TPU" if not any(off.values()) else f"; NOT on the TPU: {off}"))
    else:
        log("coral: no edgetpu_compiler (give --edgetpu-compiler): the *_edgetpu.tflite models are missing, the board runs the CPU int8 models only")
    json.dump(res, open(os.path.join(out, "checks.json"), "w"), indent=1, default=float)
    return res
