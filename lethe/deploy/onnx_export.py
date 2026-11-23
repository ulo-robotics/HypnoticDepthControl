#!/usr/bin/env python
"""
Export a trained policy as an ONNX model of one control step (the math of cpp/lethe_infer.hpp and policy.act, batched over patients), and
from it reduced-precision models and engines for other runtimes. Every model is checked in its own runtime, in closed loops on the numpy
reference plant over every validation scenario, against the float32 model.

  python deploy/onnx_export.py runs/lethe/best_model.pkl --out runs/lethe/onnx                        # policy.onnx, policy_int8.onnx (onnxruntime)
  python deploy/onnx_export.py runs/lethe/best_model.pkl --out runs/lethe/onnx --tensorrt fp32,fp16,int8      # + policy_<p>.plan
  python deploy/onnx_export.py runs/lethe/best_model.pkl --out runs/lethe/onnx --litert fp32,dynamic          # + policy_<p>.tflite

policy.onnx (opset 18, float32, batch B dynamic). Inputs: y (B) measured BIS, tgt / tgt_prev (B) the
reference now / at the previous step, u_prev (B) the previous infusion, phys_re / phys_im (B, modes) the physics compensator's modes,
s4d_re / s4d_im (L, B, H, NS) the S4D states (all zero at the start of an episode). Outputs: u (B) the infusion (mg/min, in [0, 100]) and
the new phys_re, phys_im, s4d_re, s4d_im. The matrices are the float64 discretization of deploy/cpp.py, rounded to float32. Nodes are named
by region: in/ (the features), net/ (encoder, S4D readout, mixing, layer norms, head), rec/ (the S4D and physics recurrences, the infusion).

Precision. The infusion is a high-gain function of the measurement (du/dBIS 3.6 mg/min per point at the median, up to 40), and a rounding
anywhere in the network acts like a small change of the measured BIS, every layer about alike: float32 alone moves u by 1e-3 mg/min from
the float64 model, TF32 matmuls (10-bit mantissas, the GPU default) by up to 1 mg/min. So the checks measure what matters clinically (the
selection score, the BIS trajectories, the infusions) and set it beside a physical floor: BIS read as an integer, as monitors display it.
  fp16    policy_fp16.onnx (TensorRT, strongly typed): the network in float16, the features, the recurrences and the infusion in float32.
  int8    policy_int8.onnx, quantize / dequantize nodes (onnxruntime, TensorRT): int8 mixing and head matmuls, weights per output channel and
          activations per tensor, the activations smoothed per channel first (x / s with diag(s) W; s = max|x|^0.75 / max|W|^0.25, as in
          SmoothQuant) and their scales fitted to the least squared error on calibration episodes (patients of their own keys). The
          encoder (4 features), the recurrences and the infusion stay float32.
  dynamic LiteRT: int8 weights, activations quantized at run time.
Recurrent states in integers do not hold up: an int8 S4D state breaks the controller (score +17), and a full-integer LiteRT model (16x8:
int16 activations and states), even with its state magnitudes equalized, drifts to twice the floor's BIS error over an episode.
"""
import argparse, os, shutil, subprocess, tempfile, time
import numpy as np

from lethe.deploy import cpp as export_cpp

UMAX, E0, EMAX, GAMMA, C50N = 100.0, 100.0, 100.0, 2.0, 4.5
INPUTS = ("y", "tgt", "tgt_prev", "u_prev", "phys_re", "phys_im", "s4d_re", "s4d_im")
OUTPUTS = ("u", "phys_re_out", "phys_im_out", "s4d_re_out", "s4d_im_out")
QUANT = ("mix0", "mix1", "mix2", "head")                           # the int8 matmuls
ALPHA = 0.75                                                       # smoothing exponent (int8)


def build(params, quant=None, taps=None, half=False):
    """the onnx.ModelProto of one control step. quant: matmul group -> (s, a), int8 with smoothing s (per input channel) and activation
    scale a. half: the net/ region in float16 (casts where it meets the float32 regions). taps: filled with matmul group -> the name of its
    activation operand (for calibration)"""
    import onnx
    from onnx import TensorProto, helper as oh, numpy_helper as onh
    d, a = export_cpp.discretized(params)
    H, NS, L, HD, P = d["H"], d["NS"], d["L"], d["head"], d["modes"]
    nodes, inits, n, reg = [], [], [0], ["in"]
    kind_of = {x: np.float32 for x in INPUTS}                      # tensor -> np.float32 / np.float16 / None (not floating point)
    region_dt = lambda: np.float16 if half and reg[0] == "net" else np.float32

    def const(v, dtype=None):
        dtype = dtype or region_dt()
        name = f"c{len(inits)}"
        inits.append(onh.from_array(np.asarray(v, dtype), name))
        kind_of[name] = dtype if dtype in (np.float32, np.float16) else None
        return name

    def cast(x, dt):
        n[0] += 1
        out = f"{reg[0]}_cast{n[0]}"
        to = TensorProto.FLOAT16 if dt is np.float16 else TensorProto.FLOAT
        nodes.append(oh.make_node("Cast", [x], [out], name=f"{reg[0]}/cast{n[0]}", to=to))
        kind_of[out] = dt
        return out

    def op(kind, *ins, **attrs):
        want = region_dt()
        ins = [cast(x, want) if kind_of.get(x) not in (None, want) else x for x in ins]
        n[0] += 1
        out = f"{reg[0]}_{kind.lower()}{n[0]}"
        nodes.append(oh.make_node(kind, list(ins), [out], name=f"{reg[0]}/{kind.lower()}{n[0]}", **attrs))
        kind_of[out] = None if kind == "QuantizeLinear" else want if any(kind_of.get(x) for x in ins) else None
        return out

    i64 = {}

    def _axes(ax):                                                 # an int64 initializer (axes, indices, slice bounds), shared
        key = tuple(ax)
        if key not in i64:
            i64[key] = f"ax{len(i64)}"
            inits.append(onh.from_array(np.asarray(ax, np.int64), i64[key]))
        return i64[key]

    def matmul(x, W, g):
        if taps is not None:
            taps[g] = x
        if not quant or g not in quant:
            return op("MatMul", x, const(W))
        s, sa = quant[g]
        z8 = const(0, np.int8)                                     # (int8 with float32 scales: not with half)
        xq = op("DequantizeLinear", op("QuantizeLinear", op("Mul", x, const(1.0 / s)), const(sa, np.float32), z8), const(sa, np.float32), z8)
        Ws = W * s[:, None]
        ws = np.maximum(np.abs(Ws).max(0), 1e-12) / 127.0
        Wq = np.clip(np.round(Ws / ws), -127, 127)
        return op("MatMul", xq, op("DequantizeLinear", const(Wq, np.int8), const(ws, np.float32), const(np.zeros(len(ws)), np.int8), axis=1))

    def wref(t):                                                   # C50N ((E0 - t) / EMAX / (1 - ...))^(1 / GAMMA)
        s = op("Div", op("Sub", const(E0), t), const(EMAX))
        return op("Mul", const(C50N), op("Pow", op("Div", s, op("Sub", const(1.0), s)), const(1.0 / GAMMA)))

    def gelu(x):                                                   # tanh approximation, as jax.nn.gelu
        x3 = op("Mul", op("Mul", x, x), x)
        t = op("Tanh", op("Mul", const(0.7978845608028654), op("Add", x, op("Mul", const(0.044715), x3))))
        return op("Mul", op("Mul", const(0.5), x), op("Add", const(1.0), t))

    def unsq(x, ax):
        return op("Unsqueeze", x, _axes(ax))

    wt = wref("tgt")
    r = op("Div", wt, const(C50N))
    rho = unsq(op("Div", wref("tgt_prev"), wt), (1,))
    s = op("Clip", op("Div", op("Sub", const(E0), "y"), const(EMAX)), const(1e-3), const(1.0 - 1e-3))
    w = op("Min", op("Mul", const(C50N), op("Pow", op("Div", s, op("Sub", const(1.0), s)), const(1.0 / GAMMA))), const(3.0 * C50N))
    f = [op("Div", op("Sub", "y", "tgt"), const(50.0)), op("Div", "u_prev", op("Mul", r, const(UMAX))),
         op("Sub", op("Div", w, op("Mul", r, const(C50N))), const(1.0))]
    f = op("Concat", *[unsq(v, (1,)) for v in f], axis=1)
    reg[0] = "rec"
    pr, pi = op("Mul", "phys_re", rho), op("Mul", "phys_im", rho)
    reg[0] = "net"
    x = op("Add", matmul(f, a["enc_W"], "enc"), const(a["enc_b"]))  # (B, H)
    new_re, new_im = [], []
    for l in range(L):
        g = lambda k: const(a[f"l{l}_{k}"])
        reg[0] = "rec"
        hr = op("Squeeze", op("Gather", "s4d_re", _axes((l,)), axis=0), _axes((0,)))
        hi = op("Squeeze", op("Gather", "s4d_im", _axes((l,)), axis=0), _axes((0,)))
        u3 = unsq(x, (2,))                                         # (B, H, 1)
        nr = op("Add", op("Sub", op("Mul", g("dA_re"), hr), op("Mul", g("dA_im"), hi)), op("Mul", g("dB_re"), u3))
        ni = op("Add", op("Add", op("Mul", g("dA_re"), hi), op("Mul", g("dA_im"), hr)), op("Mul", g("dB_im"), u3))
        new_re.append(unsq(nr, (0,)))
        new_im.append(unsq(ni, (0,)))
        acc = op("ReduceSum", op("Sub", op("Mul", nr, g("C_re")), op("Mul", ni, g("C_im"))), _axes((2,)), keepdims=0)
        reg[0] = "net"
        act = gelu(op("Add", op("Mul", const(2.0), acc), op("Mul", g("D"), x)))
        z = op("Add", matmul(act, a[f"l{l}_W_mix"], f"mix{l}"), g("b_mix"))                       # (B, 2H)
        zv = op("Slice", z, _axes((0,)), _axes((H,)), _axes((1,)))
        zg = op("Slice", z, _axes((H,)), _axes((2 * H,)), _axes((1,)))
        x = op("Add", x, op("Mul", zv, op("Sigmoid", zg)))
        x = op("Sub", x, op("ReduceMean", x, _axes((1,)), keepdims=1))
        inv = op("Reciprocal", op("Sqrt", op("Add", op("ReduceMean", op("Mul", x, x), _axes((1,)), keepdims=1), const(1e-5))))
        x = op("Add", op("Mul", op("Mul", x, inv), g("ln_g")), g("ln_b"))
    if HD:
        hid = gelu(op("Add", matmul(x, a["head_W1"], "head"), const(a["head_b1"])))
        o = op("Add", op("ReduceSum", op("Mul", hid, const(a["dec_W"])), _axes((1,)), keepdims=0), const(a["dec_b"]))
    else:
        o = op("Add", op("ReduceSum", op("Mul", x, const(a["dec_W"])), _axes((1,)), keepdims=0), const(a["dec_b"]))
    reg[0] = "rec"
    ul = op("Add", const(a["u0"]), op("ReduceSum", op("Sub", op("Mul", const(a["Cout_re"]), pr), op("Mul", const(a["Cout_im"]), pi)),
                                      _axes((1,)), keepdims=0))
    u = op("Clip", op("Mul", r, op("Add", ul, op("Mul", const(10.0), o))), const(0.0), const(UMAX))
    un, wn = unsq(op("Div", u, r), (1,)), unsq(op("Div", w, r), (1,))
    lr, li = const(a["lam_re"]), const(a["lam_im"])
    drive = lambda p: op("Add", op("Mul", const(a[f"Bu_{p}"]), un), op("Mul", const(a[f"Bw_{p}"]), wn))
    pr2 = op("Add", op("Sub", op("Mul", lr, pr), op("Mul", li, pi)), drive("re"))
    pi2 = op("Add", op("Add", op("Mul", lr, pi), op("Mul", li, pr)), drive("im"))
    nodes += [oh.make_node("Identity", [u], ["u"], name="rec/u"), oh.make_node("Identity", [pr2], ["phys_re_out"], name="rec/phys_re"),
              oh.make_node("Identity", [pi2], ["phys_im_out"], name="rec/phys_im"),
              oh.make_node("Concat", new_re, ["s4d_re_out"], axis=0, name="rec/s4d_re"),
              oh.make_node("Concat", new_im, ["s4d_im_out"], axis=0, name="rec/s4d_im")]
    F = TensorProto.FLOAT
    vec = lambda name: oh.make_tensor_value_info(name, F, ["B"])
    ins = [vec("y"), vec("tgt"), vec("tgt_prev"), vec("u_prev"), oh.make_tensor_value_info("phys_re", F, ["B", P]),
           oh.make_tensor_value_info("phys_im", F, ["B", P]), oh.make_tensor_value_info("s4d_re", F, [L, "B", H, NS]),
           oh.make_tensor_value_info("s4d_im", F, [L, "B", H, NS])]
    outs = [vec("u"), oh.make_tensor_value_info("phys_re_out", F, ["B", P]), oh.make_tensor_value_info("phys_im_out", F, ["B", P]),
            oh.make_tensor_value_info("s4d_re_out", F, [L, "B", H, NS]), oh.make_tensor_value_info("s4d_im_out", F, [L, "B", H, NS])]
    graph = oh.make_graph(nodes, "lethe_step", ins, outs, inits)
    m = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], producer_name="lethe deploy/onnx_export.py")
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m, d


# ---------------------------------------------------------------- closed loops of an engine with the ONNX interface
def _slice(feed, i, j):
    return {k: v[:, i:j] if k.startswith("s4d") else v[i:j] for k, v in feed.items()}


def loop(run, pts, d, maxb=None, y_int=False, keep=None):
    """closed loop on the numpy plant of an engine with the ONNX interface (run(feed) -> outputs in OUTPUTS order), maxb patients per call;
    y_int: BIS read as an integer; keep(k, feed): sees every step's inputs -> BIS, infusion (B, N)"""
    from lethe.core import model as M
    B = pts["mu"].shape[0]
    plant = M.Plant(pts)
    z = lambda *s: np.zeros(s, np.float32)
    st, up = [z(B, d["modes"]), z(B, d["modes"]), z(d["L"], B, d["H"], d["NS"]), z(d["L"], B, d["H"], d["NS"])], z(B)
    for k in range(plant.N):
        y = plant.measure()
        feed = dict(y=(np.round(y) if y_int else y).astype(np.float32), tgt=pts["tgt"][:, k].astype(np.float32),
                    tgt_prev=pts["tgt"][:, max(k - 1, 0)].astype(np.float32), u_prev=up, **dict(zip(INPUTS[4:], st)))
        if keep:
            keep(k, feed)
        if maxb is None or maxb >= B:
            out = run(feed)
        else:
            parts = [run(_slice(feed, i, i + maxb)) for i in range(0, B, maxb)]
            out = [np.concatenate([p[j] for p in parts], 1 if j >= 3 else 0) for j in range(5)]
        up, st = np.asarray(out[0], np.float32).reshape(B), [np.asarray(v, np.float32) for v in out[1:]]
        plant.step(np.clip(up.astype(np.float64), 0.0, UMAX))
    return plant.hist.T, plant.us.T


def scenarios(n, seed):
    """n patients of every validation scenario, scenario i from key seed + i"""
    import jax
    from lethe.core import model as M
    return {name: M.to_numpy(M.sample_patients(jax.random.PRNGKey(seed + i), n, sp)) for i, (name, (sp, _)) in enumerate(M.VAL.items())}


def clinical(runs, sc):
    """the selection score of closed loops {scenario: (BIS, u)}"""
    from lethe.core import model as M
    return M.score({n: {k: float(np.mean(v)) for k, v in M.metrics(b, u, sc[n]["tgt"]).items()} for n, (b, u) in runs.items()})


def compare(runs, ref, sc):
    """closed loops vs the reference's: score difference, p99 / max |du| (mg/min), p99 / max |dBIS|"""
    du = np.concatenate([np.abs(runs[n][1] - ref[n][1]).ravel() for n in sc])
    db = np.concatenate([np.abs(runs[n][0] - ref[n][0]).ravel() for n in sc])
    return clinical(runs, sc) - clinical(ref, sc), np.percentile(du, 99), du.max(), np.percentile(db, 99), db.max()


def against_jax(run, params, d, pts):
    """the step vs the JAX policy (policy.act, float32), each in its own closed loop: max |du|"""
    import jax
    from lethe.core import model as M
    from lethe.core import policy as P
    jax.config.update("jax_default_matmul_precision", "highest")
    ref, plant = P.PolicyController(params), M.Plant(pts)
    ref.reset(pts, plant)
    for k in range(plant.N):
        plant.step(np.clip(ref.act(k, plant.measure()), 0.0, M.UMAX))
    _, u = loop(run, pts, d)
    return float(np.max(np.abs(u - plant.us.T)))


def ort_runner(model):
    import onnxruntime as ort
    s = ort.InferenceSession(model if isinstance(model, (str, bytes)) else model.SerializeToString(), providers=["CPUExecutionProvider"])
    return lambda feed: s.run(None, feed)


# ---------------------------------------------------------------- int8 calibration
def calibrate(params, d, n, seed=1000):
    """closed loops of the float32 step (onnxruntime) on n patients of every scenario, keys seed + i: the int8 matmuls' smoothing and
    activation scales, group -> (s, a)"""
    from onnx import TensorProto, helper as oh
    taps = {}
    m, _ = build(params, taps=taps)
    m.graph.output.extend(oh.make_tensor_value_info(t, TensorProto.FLOAT, None) for t in taps.values())
    run0 = ort_runner(m)
    ops = {g: [] for g in taps}

    def run(feed):
        out = run0(feed)
        for g, v in zip(taps, out[5:]):
            ops[g].append(v[::4])
        return out[:5]
    for pts in scenarios(n, seed).values():
        loop(run, pts, d)
    _, a = export_cpp.discretized(params)
    W = {f"mix{l}": a[f"l{l}_W_mix"] for l in range(d["L"])}
    if d["head"]:
        W["head"] = a["head_W1"]
    quant = {}
    for g, Wg in W.items():
        x = np.concatenate(ops[g]).astype(np.float64)
        s = (np.abs(x).max(0) + 1e-9) ** ALPHA / (np.abs(Wg).max(1) + 1e-9) ** (1 - ALPHA)
        xs = x / s
        m_ = np.abs(xs).max()
        err = lambda sc_: np.mean((np.clip(np.round(xs / sc_), -127, 127) * sc_ - xs) ** 2)
        quant[g] = (s, min((m_ * f / 127.0 for f in np.linspace(0.3, 1.0, 71)), key=err))
    return quant


# ---------------------------------------------------------------- TensorRT
class TensorRT:
    """an engine built with the TensorRT Python API from an ONNX file as a strongly typed network (each layer in the type of its ONNX
    tensors: policy.onnx float32 without TF32, policy_fp16.onnx, policy_int8.onnx), batch 1 .. maxb; run through cuda-python"""

    def __init__(self, path, d, plan, maxb=1024, optb=64):
        try:
            import tensorrt as trt
        except ImportError:
            import tensorrt_bindings as trt
        try:
            from cuda.bindings import runtime as cr
        except ImportError:
            from cuda import cudart as cr
        self.cr, self.d = cr, d
        log = trt.Logger(trt.Logger.ERROR)
        b = trt.Builder(log)
        net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        ps = trt.OnnxParser(net, log)
        if not ps.parse(open(path, "rb").read()):
            raise RuntimeError("TensorRT could not parse the model: " + "; ".join(str(ps.get_error(i)) for i in range(ps.num_errors)))
        cfg = b.create_builder_config()
        cfg.clear_flag(trt.BuilderFlag.TF32)
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        prof = b.create_optimization_profile()
        for i in range(net.num_inputs):
            t = net.get_input(i)
            shp = lambda m: tuple(m if s_ == -1 else s_ for s_ in t.shape)
            prof.set_shape(t.name, shp(1), shp(optb), shp(maxb))
        cfg.add_optimization_profile(prof)
        ser = b.build_serialized_network(net, cfg)
        if ser is None:
            raise RuntimeError(f"TensorRT build failed ({path})")
        open(plan, "wb").write(ser)
        self.eng = trt.Runtime(log).deserialize_cuda_engine(ser)
        self.ctx = self.eng.create_execution_context()
        info = self.eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
        self.formats = {k: info.count(k) for k in ("Half", "Int8")}
        self.bufs, self.B = {}, None

    def _shape(self, n, B):
        d = self.d
        return (d["L"], B, d["H"], d["NS"]) if n.startswith("s4d") else (B, d["modes"]) if n.startswith("phys") else (B,)

    def __call__(self, feed):
        cr = self.cr
        B = len(feed["y"])
        if B != self.B:
            for p in self.bufs.values():
                cr.cudaFree(p)
            self.bufs = {n: cr.cudaMalloc(int(np.prod(self._shape(n, B))) * 4)[1] for n in INPUTS + OUTPUTS}
            for n, p in self.bufs.items():
                self.ctx.set_tensor_address(n, p)
                if n in INPUTS:
                    self.ctx.set_input_shape(n, self._shape(n, B))
            self.B = B
        for n in INPUTS:
            v = np.ascontiguousarray(feed[n], np.float32)
            cr.cudaMemcpy(self.bufs[n], v.ctypes.data, v.nbytes, cr.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.ctx.execute_async_v3(0)
        out = [np.empty(self._shape(n, B), np.float32) for n in OUTPUTS]
        for n, v in zip(OUTPUTS, out):
            cr.cudaMemcpy(v.ctypes.data, self.bufs[n], v.nbytes, cr.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return out

    def latency(self, feed, reps=200):
        """median wall time of one call (copies included), ms"""
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            self(feed)
            ts.append(time.perf_counter() - t0)
        return 1e3 * float(np.median(ts))


# ---------------------------------------------------------------- LiteRT
def litert_convert(onnx_path, out, d, modes):
    """policy_<mode>.tflite at batch 1 from a TensorFlow SavedModel of the ONNX model (onnx2tf), modes fp32 / dynamic (int8 weights)"""
    import tensorflow as tf
    L, H, NS, P = d["L"], d["H"], d["NS"], d["modes"]
    tmp = tempfile.mkdtemp(dir=out)
    shutil.copy(onnx_path, os.path.join(tmp, "policy.onnx"))      # (onnx2tf rewrites its input with the fixed shapes)
    one = {"phys_re": f"1,{P}", "phys_im": f"1,{P}", "s4d_re": f"{L},1,{H},{NS}", "s4d_im": f"{L},1,{H},{NS}"}
    cmd = ["onnx2tf", "-i", os.path.join(tmp, "policy.onnx"), "-o", os.path.join(tmp, "tf"), "-tb", "tf_converter", "-osd",
           "-kat", *INPUTS, "-ois", *[f"{x}:{one.get(x, '1')}" for x in INPUTS]]             # -kat: no NHWC transposes
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    sm = os.path.join(tmp, "tf")
    if res.returncode != 0 or not os.path.exists(os.path.join(sm, "saved_model.pb")):
        raise RuntimeError("onnx2tf failed:\n" + res.stdout[-3000:] + res.stderr[-2000:])
    paths = {}
    for mode in modes:
        c = tf.lite.TFLiteConverter.from_saved_model(sm)
        if mode == "dynamic":
            c.optimizations = [tf.lite.Optimize.DEFAULT]
        paths[mode] = os.path.join(out, f"policy_{mode}.tflite")
        open(paths[mode], "wb").write(c.convert())
    shutil.rmtree(tmp)
    return paths


def litert_runner(path, d):
    """a LiteRT interpreter as an engine with the ONNX interface (batch 1); the SavedModel's outputs are output_0 .. output_4 in OUTPUTS order"""
    from ai_edge_litert.interpreter import Interpreter
    run = Interpreter(model_path=path).get_signature_runner()
    shp = [(1,), (1, d["modes"]), (1, d["modes"]), (d["L"], 1, d["H"], d["NS"]), (d["L"], 1, d["H"], d["NS"])]

    def step(feed):
        r = run(**feed)
        return [r[f"output_{i}"].reshape(s) for i, s in enumerate(shp)]
    return step


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=64, help="patients per validation scenario in the checks (and in the calibration)")
    ap.add_argument("--tensorrt", default="", help="comma-separated: fp32, fp16, int8 (TensorRT Python API and cuda-python)")
    ap.add_argument("--litert", default="", help="comma-separated: fp32, dynamic (onnx2tf, TensorFlow, ai-edge-litert)")
    ap.add_argument("--no-int8", dest="int8", action="store_false", help="do not write policy_int8.onnx")
    a = ap.parse_args()
    import onnx
    from lethe.core import policy as Pol
    os.makedirs(a.out, exist_ok=True)
    params = Pol.load_model(a.model)["params"]
    m, d = build(params)
    path = os.path.join(a.out, "policy.onnx")
    onnx.save(m, path)
    print(f"wrote {path} (H {d['H']}, NS {d['NS']}, L {d['L']}, head {d['head']}, modes {d['modes']}; {len(m.graph.node)} nodes)", flush=True)
    trt_p, lite_p = [p for p in a.tensorrt.split(",") if p], [p for p in a.litert.split(",") if p]
    run32 = ort_runner(path)
    sc = scenarios(a.patients, seed=0)
    print(f"onnxruntime float32 vs the JAX policy (float32), closed loop, {a.patients} mixed_hard patients: max |du| "
          f"{against_jax(run32, params, d, sc['mixed_hard']):.1e} mg/min", flush=True)
    ref = {n: loop(run32, pts, d) for n, pts in sc.items()}
    print(f"\nclosed loops, {a.patients} patients x {len(sc)} validation scenarios, against onnxruntime float32 "
          f"(score {clinical(ref, sc):.4f}):")
    print(f"  {'':40s} {'dscore':>9s} {'|du| p99':>9s} {'max':>7s} {'|dBIS| p99':>10s} {'max':>6s}")
    show = lambda name, runs, extra="": print(f"  {name:40s} {'%+9.1e  %8.2e %7.2f %10.3f %6.2f' % compare(runs, ref, sc)}{extra}", flush=True)
    show("floor: BIS read as an integer", {n: loop(run32, pts, d, y_int=True) for n, pts in sc.items()})
    onnx_of = {"fp32": path}
    if "fp16" in trt_p:
        onnx_of["fp16"] = os.path.join(a.out, "policy_fp16.onnx")
        onnx.save(build(params, half=True)[0], onnx_of["fp16"])
    if a.int8 or "int8" in trt_p:
        onnx_of["int8"] = os.path.join(a.out, "policy_int8.onnx")
        onnx.save(build(params, quant=calibrate(params, d, a.patients))[0], onnx_of["int8"])
        r8 = ort_runner(onnx_of["int8"])
        show("onnxruntime int8 (policy_int8.onnx)", {n: loop(r8, pts, d) for n, pts in sc.items()})
    for p in trt_p:
        plan = os.path.join(a.out, f"policy_{p}.plan")
        eng = TensorRT(onnx_of[p], d, plan)
        lat = eng.latency(_feed0(d, a.patients))
        show(f"TensorRT {p} (policy_{p}.plan)", {n: loop(eng, pts, d) for n, pts in sc.items()},
             f"   {lat:.3f} ms per step of {a.patients} patients, copies included; "
             f"layers: {eng.formats['Half']} fp16, {eng.formats['Int8']} int8")
    lite = litert_convert(path, a.out, d, lite_p) if lite_p else {}
    for mode, lp in lite.items():
        show(f"LiteRT {mode} (policy_{mode}.tflite)", {n: loop(litert_runner(lp, d), pts, d, maxb=1) for n, pts in sc.items()},
             f"   {os.path.getsize(lp) / 1024:.0f} KiB")


def _feed0(d, B):
    z = lambda *s: np.zeros(s, np.float32)
    return dict(y=np.full(B, 50, np.float32), tgt=np.full(B, 50, np.float32), tgt_prev=np.full(B, 50, np.float32), u_prev=z(B),
                phys_re=z(B, d["modes"]), phys_im=z(B, d["modes"]), s4d_re=z(d["L"], B, d["H"], d["NS"]), s4d_im=z(d["L"], B, d["H"], d["NS"]))


if __name__ == "__main__":
    main()
