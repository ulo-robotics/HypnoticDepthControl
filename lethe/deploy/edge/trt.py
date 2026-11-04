"""
The ONNX step (lethe.deploy.onnx_export) for TensorRT 8.x's parser (Jetson Nano, JetPack 4.6, TensorRT 8.2): opset 13, IR 7, batch fixed
to 1. Of this graph's operators only ReduceMean changed between opsets 13 and 18 (its axes are an input since 18, an attribute before).
export() writes the converted model and checks it with onnxruntime against the opset-18 model and against the JAX policy's recorded
closed loops (lethe.deploy.cpp.write_trace): the recorded measurements and targets are replayed, the states fed back.
"""
import struct
import numpy as np

STATES = ("phys_re", "phys_im", "s4d_re", "s4d_im")


def to_opset13(m):
    import onnx
    from onnx import helper, numpy_helper
    m = onnx.ModelProto.FromString(m.SerializeToString())
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    used = set()
    for nd in m.graph.node:
        if nd.op_type == "ReduceMean" and len(nd.input) > 1:
            axes = init[nd.input[1]]
            used.add(nd.input[1])
            del nd.input[1]
            nd.attribute.append(helper.make_attribute("axes", [int(a) for a in np.ravel(axes)]))
    still = {i for nd in m.graph.node for i in nd.input}
    keep = [t for t in m.graph.initializer if t.name not in used or t.name in still]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = 13
    for v in list(m.graph.input) + list(m.graph.output):              # batch 1
        for dim in v.type.tensor_type.shape.dim:
            if dim.dim_param:
                dim.ClearField("dim_param")
                dim.dim_value = 1
    m.ir_version = 7
    onnx.checker.check_model(m)
    return m


def read_trace(path):
    with open(path, "rb") as f:
        n, N = struct.unpack("<2i", f.read(8))
        return np.frombuffer(f.read(), "<f8").reshape(n, 3, N)


def replay(sess, tr):
    """one patient's recorded measurements / targets through an onnxruntime session of the step, batch 1 -> u (N,)"""
    shp = {x.name: x.shape for x in sess.get_inputs()}
    st = {k: np.zeros([s if isinstance(s, int) else 1 for s in shp[k]], np.float32) for k in STATES}
    up, us = np.zeros(1, np.float32), []
    for k in range(tr.shape[1]):
        feed = dict(y=np.float32([tr[0, k]]), tgt=np.float32([tr[1, k]]), tgt_prev=np.float32([tr[1, max(k - 1, 0)]]), u_prev=up, **st)
        o = sess.run(None, feed)
        up = np.asarray(o[0], np.float32).reshape(1)
        st = dict(zip(STATES, o[1:]))
        us.append(float(up[0]))
    return np.array(us)


def export(params, out, trace):
    import onnx
    import onnxruntime as ort
    from lethe.deploy import onnx_export
    m18 = onnx_export.build(params)[0]
    m13 = to_opset13(m18)
    onnx.save(m13, out)
    S = lambda m: ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    s18, s13 = S(m18), S(m13)
    tr = read_trace(trace)
    d = e = 0.0
    for t in tr:
        a, b = replay(s18, t), replay(s13, t)
        d, e = max(d, np.abs(a - b).max()), max(e, np.abs(b - t[2]).max())
    return dict(path=out, vs_opset18=float(d), vs_jax=float(e))         # max |du|, mg/min
