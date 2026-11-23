"""closed loops of the policy running on emulated ESP32s (QEMU, UART over tcp), one patient at a time on the numpy plant, against
onnxruntime float32 run the same way: python hil.py model.pkl model.bin patients host:port:mode ...  (mode F float / D double)"""
import ctypes, os, queue, socket, struct, subprocess, sys, threading, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))                # the repository
from lethe.core import policy as P
from lethe.deploy import onnx_export as E


def one(pts, i, B):
    o = {}
    for k, v in pts.items():
        v = np.asarray(v)
        if v.ndim and v.shape[0] == B:
            o[k] = v[i:i + 1]
        elif v.ndim and v.shape[-1] == B:
            o[k] = v[..., i:i + 1]
        else:
            raise ValueError(f"{k} {v.shape}")
    return o


def wrap(d, f):
    """an engine with the ONNX interface around a batch-1 step f(k, y, tgt, tgt_prev) -> u that keeps its own state; one per episode (k
    counts its calls: k = 0 resets the device)"""
    z = [np.zeros((1, d["modes"]), np.float32)] * 2 + [np.zeros((d["L"], 1, d["H"], d["NS"]), np.float32)] * 2
    k = [0]

    def run(feed):
        u = f(k[0], float(feed["y"][0]), float(feed["tgt"][0]), float(feed["tgt_prev"][0]))
        k[0] += 1
        return [np.array([u], np.float32)] + z
    return run


class Dev:
    def __init__(self, host, port, mode):
        for _ in range(600):
            try:
                self.s = socket.create_connection((host, port)); break
            except OSError:
                time.sleep(0.5)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        buf = b""
        while not buf.endswith(b"\n") or b"LETHE READY" not in buf:
            buf += self.s.recv(1)
        self.s.sendall(mode.encode())
        if self.rx(1) != b"K":
            raise RuntimeError(f"{host}:{port} {mode}: policy does not fit")

    def rx(self, n):
        b = b""
        while len(b) < n:
            b += self.s.recv(n - len(b))
        return b

    def __call__(self, k, y, t, tp):
        self.s.sendall(b"S" + struct.pack("<ifff", k, y, t, tp))
        return struct.unpack("<f", self.rx(4))[0]


def native(path, dbl):
    lib = "/tmp/liblethe.so"
    if not os.path.exists(lib):
        subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-I" + os.path.join(os.path.dirname(os.path.dirname(HERE)), "cpp"),
                        os.path.join(HERE, "native.cpp"), "-o", lib], check=True)
    L = ctypes.CDLL(lib)
    L.lethe_new.restype = ctypes.c_void_p
    L.lethe_step.restype = ctypes.c_float
    L.lethe_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    h = L.lethe_new(path.encode(), int(dbl))
    return lambda k, y, t, tp: L.lethe_step(h, k, y, t, tp)


def run_all(steps, sc, d, **kw):
    """every patient of every scenario through one of the batch-1 steps (one thread each) -> {scenario: (BIS (B, N), u (B, N))}"""
    jobs = queue.Queue()
    for n, pts in sc.items():
        for i in range(pts["mu"].shape[0]):
            jobs.put((n, i))
    res = {}

    def work(f):
        while True:
            try:
                n, i = jobs.get_nowait()
            except queue.Empty:
                return
            B = sc[n]["mu"].shape[0]
            res[n, i] = E.loop(wrap(d, f), one(sc[n], i, B), d, **kw)
    th = [threading.Thread(target=work, args=(f,)) for f in steps]
    [t.start() for t in th]
    [t.join() for t in th]
    return {n: tuple(np.concatenate([res[n, i][j] for i in range(pts["mu"].shape[0])], 0) for j in range(2)) for n, pts in sc.items()}


def main():
    pkl, mbin, npat = sys.argv[1], sys.argv[2], int(sys.argv[3])
    devs = {}
    for a in sys.argv[4:]:
        host, port, mode = a.split(":")
        devs.setdefault(f"{host.split('_', 1)[1]} {'float' if mode == 'F' else 'double'}", []).append((host, int(port), mode))
    params = P.load_model(pkl)["params"]
    m, d = E.build(params)
    sess = E.ort_runner(m)
    sc = E.scenarios(npat, seed=0)
    t0 = time.time()

    def ort_step():
        st = {}

        def f(k, y, t, tp):
            if k == 0:
                st.clear()
                s4 = np.zeros((d["L"], 1, d["H"], d["NS"]), np.float32)
                st.update(u=np.zeros(1, np.float32), s=[np.zeros((1, d["modes"]), np.float32)] * 2 + [s4] * 2)
            feed = dict(y=np.full(1, y, np.float32), tgt=np.full(1, t, np.float32), tgt_prev=np.full(1, tp, np.float32),
                        u_prev=st["u"], **dict(zip(E.INPUTS[4:], st["s"])))
            o = sess(feed)
            st["u"], st["s"] = np.asarray(o[0], np.float32), [np.asarray(v, np.float32) for v in o[1:]]
            return float(o[0][0])
        return f
    ref = run_all([ort_step()], sc, d)
    print(f"closed loops, {npat} patients x {len(sc)} validation scenarios, one patient at a time, against onnxruntime float32 "
          f"(score {E.clinical(ref, sc):.4f}; {time.time() - t0:.0f} s):")
    print(f"  {'':34s} {'dscore':>9s} {'|du| p99':>9s} {'max':>7s} {'|dBIS| p99':>10s} {'max':>6s}")
    show = lambda name, runs, extra="": print(f"  {name:34s} {'%+9.1e  %8.2e %7.2f %10.3f %6.2f' % E.compare(runs, ref, sc)}{extra}", flush=True)
    show("floor: BIS read as an integer", run_all([ort_step()], sc, d, y_int=True))
    for dbl in (0, 1):
        t0 = time.time()
        show(f"C++ {'double' if dbl else 'float'}, x86-64 (glibc)", run_all([native(mbin, dbl)], sc, d), f"   {time.time() - t0:.0f} s")
    for name, eps in devs.items():
        t0 = time.time()
        try:
            steps = [Dev(*e) for e in eps]
        except RuntimeError as e:
            print(f"  {name:34s} {e}")
            continue
        show(f"QEMU {name}", run_all(steps, sc, d), f"   {time.time() - t0:.0f} s, {len(steps)} instances")


if __name__ == "__main__":
    main()
