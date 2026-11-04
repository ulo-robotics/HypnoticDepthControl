"""
The edge benchmarks' command line (edge.py): the deployed controller and the classical baselines built and timed on real boards, over ssh.

  python edge.py prepare runs/lethe/best_model.pkl --out runs/lethe/edge                  # what the boards need (here, once)
  python edge.py run HOST [HOST ...] --dir runs/lethe/edge                                 # copy, build on the board, run, collect
  python edge.py table runs/lethe/edge                                                     # per-step latency of every result

prepare writes, under --out (it needs the training environment; CasADi for the baselines, onnx and onnxruntime for TensorRT, TensorFlow
for the Coral models):
  lethe/       model.bin, trace.bin (lethe.deploy.cpp: the policy and its recorded closed loops on mixed_hard patients)
  baselines/   the CasADi-generated C of UKF-MM, MPC-MM's NLP and MHE+NMPC's NLPs (lethe.deploy.edge.baselines), and each baseline's
               closed loop of the Python implementation on the same patients (trace_<name>.bin)
  tensorrt/    (--tensorrt) the policy's ONNX step for TensorRT 8.x (lethe.deploy.edge.trt: opset 13, batch 1) and trace.bin
  coral/       (--coral) the int8 TPU parts of the split policy and their Edge TPU compilations (lethe.deploy.edge.coral; the compiler
               runs on x86-64 only: --edgetpu-compiler), the discretized model and trace.bin
run needs only Python's standard library and ssh. HOST is any ssh destination (an alias, user@host; options with --ssh), or `local` for
this machine. For each host it copies cpp/, edge/ and the prepared files to --remote-dir (relative to the remote home), builds there with
the board's own compiler (g++ -O3 -march=native; the baselines link IPOPT through pkg-config; tensorrt needs TensorRT and CUDA), runs each
benchmark pinned to one core (taskset; --core auto = the last core) and appends one JSON line per benchmark to --dir/results/<host>.jsonl,
with the board's description (model, kernel, governor, compiler). Every benchmark first replays the recorded closed loops and reports the
largest difference from them (max_abs_err_vs_*), then times every control step.
  lethe      cpp/bench.cpp: the C++ engine in float and in double
  ukfmm      edge/baselines/bench_base.cpp: the generated UKF-MM step
  mpcmm      the generated UKF-MM estimator + IPOPT over the generated NLP, warm-started (lethe.baselines.mpc.MPCMM)
  mhe        edge/baselines/bench_mhe.cpp: MHE+NMPC, 7 MHE problems and one NMPC per step (lethe.baselines.mpc.MHENMPC)
  tensorrt   edge/tensorrt: engines built on the board (fp32, fp16), one step per enqueue, with and without a CUDA graph
  coral      edge/coral/bench.py: the split policy with its TPU parts on the CPU (TFLite int8) and on a Coral Edge TPU, the TPU's outputs
             compared with the CPU's on every call; needs numpy, tflite_runtime and libedgetpu on the board (--coral-python: the
             interpreter that has them)
Building the generated MPC C takes minutes (about 6 on a Raspberry Pi 5, over 20 on a Pi 4); object files are kept on the board and only
rebuilt when their source changes.
"""
import argparse, io, json, os, re, shlex, subprocess, sys, tarfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))      # the repository
WHAT = ("lethe", "ukfmm", "mpcmm", "mhe", "tensorrt", "coral")
REPEATS = dict(lethe=2000, ukfmm=2000, mpcmm=40, mhe=8, tensorrt=200, coral=50)
SOURCES = {"cpp": ("lethe_infer.hpp", "bench.cpp"), "edge/baselines": ("bench_base.cpp", "bench_mhe.cpp", "build.sh"),
           "edge/tensorrt": ("build_trt.py", "trt_bench.cpp"), "edge/coral": ("bench.py",)}
NEEDS = dict(lethe=("lethe/model.bin", "lethe/trace.bin"), ukfmm=("baselines/baselines.c", "baselines/trace_ukfmm.bin"),
             mpcmm=("baselines/baselines.c", "baselines/trace_mpcmm.bin"), mhe=("baselines/mhe_nlp.c", "baselines/trace_mhe.bin"),
             tensorrt=("tensorrt/policy_op13.onnx", "tensorrt/trace.bin"), coral=("coral/head.tflite", "coral/arrays.npz", "coral/trace.bin"))


# ---------------------------------------------------------------- prepare (training environment)
def _copy(src, dst):
    with open(src, "rb") as f, open(dst, "wb") as g:
        g.write(f.read())


def prepare(model, out, baselines=("ukfmm", "mpcmm", "mhe"), tensorrt=False, patients=8, coral=False, compiler=None, coral_patients=32):
    from lethe.core import policy as P
    from lethe.deploy import cpp
    params = P.load_model(model)["params"]
    os.makedirs(os.path.join(out, "lethe"), exist_ok=True)
    cpp.write_model(params, os.path.join(out, "lethe", "model.bin"))
    cpp.write_trace(params, os.path.join(out, "lethe", "trace.bin"), patients)
    print(f"lethe: model.bin, trace.bin ({patients} patients)", flush=True)
    if baselines:
        from lethe.deploy.edge import baselines as codegen
        b = os.path.join(out, "baselines")
        if {"ukfmm", "mpcmm"} & set(baselines):
            print(codegen.ukfmm_mpcmm(b), flush=True)
        if "mhe" in baselines:
            print(codegen.mhe_nmpc(b), flush=True)
        for name, dt in codegen.reference_traces(b, patients, baselines).items():
            print(f"{name}: reference closed loop, {patients} patients in {dt:.0f} s", flush=True)
    if tensorrt:
        from lethe.deploy.edge import trt
        t = os.path.join(out, "tensorrt")
        os.makedirs(t, exist_ok=True)
        _copy(os.path.join(out, "lethe", "trace.bin"), os.path.join(t, "trace.bin"))
        r = trt.export(params, os.path.join(t, "policy_op13.onnx"), os.path.join(t, "trace.bin"))
        print(f"tensorrt: {r['path']} (opset 13, batch 1): max |du| vs opset 18 {r['vs_opset18']:.1e}, vs the JAX trace {r['vs_jax']:.1e} "
              "mg/min", flush=True)
    if coral:
        from lethe.deploy.edge import coral as C
        c = os.path.join(out, "coral")
        C.build(params, c, coral_patients, compiler, log=lambda *m: print(*m, flush=True))
        _copy(os.path.join(out, "lethe", "trace.bin"), os.path.join(c, "trace.bin"))


# ---------------------------------------------------------------- run (standard library only)
def _ssh(host, opts):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30"] + opts + [host]


def shell(host, cmd, opts, data=None):
    """run a bash command on the host (or locally); -> (exit code, stdout, stderr)"""
    argv = ["bash", "-c", cmd] if host == "local" else _ssh(host, opts) + ["bash -c " + shlex.quote(cmd)]
    r = subprocess.run(argv, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode, r.stdout.decode(errors="replace"), r.stderr.decode(errors="replace")


def bundle(d, what):
    """tar.gz of the sources and the prepared files the benchmarks need (modification times kept, so the board rebuilds only what changed)"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for sub, names in SOURCES.items():
            for n in names:
                tf.add(os.path.join(ROOT, sub, n), arcname=os.path.join(sub.split("/")[-1], n))
        for sub in sorted({p.split("/")[0] for w in what for p in NEEDS[w]}):
            for n in sorted(os.listdir(os.path.join(d, sub))):
                if not n.endswith((".o", ".plan", ".log", ".json")) or n == "dims.json":
                    tf.add(os.path.join(d, sub, n), arcname=os.path.join(sub, n))
    return buf.getvalue()


INFO = r"""
m=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || lscpu | sed -n 's/^Model name: *//p' | head -1)
echo "model=$m"; echo "kernel=$(uname -srm)"; echo "cores=$(nproc)"; echo "cxx=$(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)"
c=%s; [ "$c" = auto ] && c=$(( $(nproc) - 1 )); echo "core=$c"
echo "governor=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null)"
echo "max_khz=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq 2>/dev/null)"
"""

BUILD = {
    "lethe": "cd cpp && { [ bench -nt bench.cpp ] && [ bench -nt lethe_infer.hpp ] || g++ -O3 -march=native -std=c++17 -o bench bench.cpp; }",
    "ukfmm": "cd baselines && { [ bench_base -nt bench_base.cpp ] && [ bench_base -nt baselines.c ] || sh build.sh bench_base.cpp baselines.c; }",
    "mhe": "cd baselines && { [ bench_mhe -nt bench_mhe.cpp ] && [ bench_mhe -nt mhe_nlp.c ] && [ bench_mhe -nt nmpc_nlp.c ] || "
           "sh build.sh bench_mhe.cpp mhe_nlp.c nmpc_nlp.c; }",
    "tensorrt": "cd tensorrt && python3 build_trt.py . fp32,fp16 && { [ trt_bench -nt trt_bench.cpp ] || g++ -O3 -std=c++14 trt_bench.cpp "
                "-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvinfer -lcudart -o trt_bench; }",
    "coral": "cd coral && {py} -c 'import numpy; from tflite_runtime.interpreter import load_delegate' 2>/dev/null || "
             "{py} -c 'import numpy; from ai_edge_litert.interpreter import load_delegate'",
}
BUILD["mpcmm"] = BUILD["ukfmm"]
RUN = {
    "lethe": ["$PIN cpp/bench lethe/model.bin lethe/trace.bin {n}"],
    "ukfmm": ["cd baselines && $PIN ./bench_base ukfmm trace_ukfmm.bin {n}"],
    "mpcmm": ["cd baselines && $PIN ./bench_base mpcmm trace_mpcmm.bin {n}"],
    "mhe": ["cd baselines && $PIN ./bench_mhe trace_mhe.bin {n}"],
    "tensorrt": [f"cd tensorrt && $PIN ./trt_bench policy_{p}.plan trace.bin {{n}} {g}" for p in ("fp32", "fp16") for g in (0, 1)],
    "coral": ["cd coral && $PIN {py} bench.py . {n}"],
}


def run_host(host, d, what, opts=(), remote="lethe_edge", core="auto", repeats=None, label=None, log=print, coral_python="python3"):
    """copy, build and run the benchmarks `what` on one host; appends to d/results/<label>.jsonl and returns the records"""
    opts, label = list(opts), label or re.sub(r"[^A-Za-z0-9_.-]", "_", host)
    for w in what:
        for p in NEEDS[w]:
            if not os.path.exists(os.path.join(d, p)):
                raise SystemExit(f"{w}: {os.path.join(d, p)} missing (python edge.py prepare ...{' --' + w if w in ('tensorrt', 'coral') else ''})")
    R = remote if os.path.isabs(remote) else "$HOME/" + remote
    code, o, e = shell(host, f"mkdir -p {R} && tar xzf - -C {R}", opts, bundle(d, what))
    if code:
        raise SystemExit(f"{host}: copy failed: {e.strip()}")
    code, o, e = shell(host, INFO % shlex.quote(str(core)), opts)
    info = dict(l.split("=", 1) for l in o.splitlines() if "=" in l)
    log(f"{host}: {info.get('model', '?')}, {info.get('cores', '?')} cores, g++ {info.get('cxx', '?')}, core {info.get('core')} "
        f"({info.get('governor') or 'no cpufreq'})")
    pin = f'PIN=""; command -v taskset >/dev/null && PIN="taskset -c {info.get("core", 0)}"; '
    os.makedirs(os.path.join(d, "results"), exist_ok=True)
    out, recs = os.path.join(d, "results", f"{label}.jsonl"), []
    for w in what:
        t0 = time.time()
        code, o, e = shell(host, f"cd {R} && " + BUILD[w].replace("{py}", coral_python), opts)
        if code:
            log(f"  {w}: build failed ({time.time() - t0:.0f} s): {e.strip()[-600:]}")
            continue
        log(f"  {w}: built in {time.time() - t0:.0f} s")
        for cmd in RUN[w]:
            n = (repeats or {}).get(w, REPEATS[w])
            code, o, e = shell(host, f"cd {R} && {pin}" + cmd.format(n=n, py=coral_python), opts)
            lines = [l for l in o.splitlines() if l.startswith("{")]
            if code or not lines:
                log(f"  {w}: run failed: {(e or o).strip()[-600:]}")
                continue
            for line in lines:                                  # one record per JSON line (the coral benchmark prints one per engine)
                rec = dict(json.loads(line), host=label, what=w, repeats=n, board=info, date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                recs.append(rec)
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                log(f"  {w}: {summary(rec)}")
    return recs


def summary(r):
    """'median / p99 us per step, max error' of one record"""
    if r["what"] == "lethe":
        return ", ".join(f"{p} {r[p]['ns_per_step_median'] / 1e3:.1f} / {r[p]['ns_per_step_p99'] / 1e3:.1f} us "
                         f"(err {r[p]['max_abs_err_vs_jax']:.1e})" for p in ("float", "double"))
    err = r.get("max_abs_err_vs_python", r.get("max_abs_err_vs_jax"))
    tag = f"{os.path.basename(r['engine'])}{' graph' if r.get('graph') else ''}: " if "engine" in r else ""
    if "max_lsb_vs_cpu_int8" in r:
        tag += f"(TPU vs CPU int8: {r['max_lsb_vs_cpu_int8']} LSB) "
    return f"{tag}{r['us_per_step_median']:.1f} / {r['us_per_step_p99']:.1f} us (mean {r['us_per_step_mean']:.1f}, err {err:.1e})"


def table(d):
    """latest record per (host, benchmark): median per-step latency in microseconds"""
    rows = {}
    for fn in sorted(os.listdir(os.path.join(d, "results"))):
        for l in open(os.path.join(d, "results", fn)):
            r = json.loads(l)
            keys = {"tensorrt": lambda: [f"tensorrt {r['engine'][7:-5]}{' graph' if r['graph'] else ''}"],
                    "coral": lambda: [f"coral {r['engine']}"]}.get(r["what"], lambda: [r["what"]])()
            if r["what"] == "lethe":
                for p in ("float", "double"):
                    rows.setdefault(r["host"], {})[f"lethe {p}"] = r[p]["ns_per_step_median"] / 1e3
            else:
                rows.setdefault(r["host"], {})[keys[0]] = r["us_per_step_median"]
    cols = sorted({c for v in rows.values() for c in v}, key=lambda c: (WHAT.index(c.split()[0]), c))
    print("| host | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for h, v in rows.items():
        print(f"| {h} | " + " | ".join(f"{v[c]:.1f}" if c in v else "" for c in cols) + " |")
    print("\nmedian microseconds per control step (5 s of patient time)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="export the policy, generate the baselines' C and their reference closed loops")
    p.add_argument("model")
    p.add_argument("--out", required=True)
    p.add_argument("--baselines", default="ukfmm,mpcmm,mhe", help="comma-separated, or '' for none")
    p.add_argument("--tensorrt", action="store_true", help="also the ONNX step for TensorRT 8.x (Jetson, JetPack 4)")
    p.add_argument("--patients", type=int, default=8, help="recorded closed loops (mixed_hard patients)")
    p.add_argument("--coral", action="store_true", help="also the Coral Edge TPU models (TensorFlow)")
    p.add_argument("--edgetpu-compiler", default=None, help="path of edgetpu_compiler (default: on the PATH)")
    p.add_argument("--coral-patients", type=int, default=32, help="patients per validation scenario in the Coral calibration and checks")
    r = sub.add_parser("run", help="copy, build and run on boards")
    r.add_argument("hosts", nargs="+", help="ssh destinations, or local")
    r.add_argument("--dir", required=True, help="the prepare output; results go to DIR/results")
    r.add_argument("--what", default="lethe,ukfmm,mpcmm,mhe", help="comma-separated of " + ", ".join(WHAT))
    r.add_argument("--ssh", default="", help="extra ssh options, e.g. '-p 2222 -i key'")
    r.add_argument("--remote-dir", default="lethe_edge", help="working directory on the board (relative to its home)")
    r.add_argument("--core", default="auto", help="core to pin to (auto: the last one)")
    r.add_argument("--repeats", default="", help="episodes to time, e.g. mpcmm=20,mhe=4 (defaults %s)" % REPEATS)
    r.add_argument("--coral-python", default="python3", help="the board's Python with numpy and tflite_runtime (coral)")
    t = sub.add_parser("table", help="median latency of the collected results")
    t.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.model, a.out, [b for b in a.baselines.split(",") if b], a.tensorrt, a.patients, a.coral, a.edgetpu_compiler,
                a.coral_patients)
    elif a.cmd == "run":
        what = [w for w in a.what.split(",") if w]
        bad = set(what) - set(WHAT)
        if bad:
            raise SystemExit(f"unknown benchmarks {sorted(bad)}; choose from {WHAT}")
        reps = {k: int(v) for k, v in (x.split("=") for x in a.repeats.split(",") if x)}
        for h in a.hosts:
            run_host(h, a.dir, what, shlex.split(a.ssh), a.remote_dir, a.core, reps, coral_python=a.coral_python)
    else:
        table(a.dir)


if __name__ == "__main__":
    sys.exit(main())
