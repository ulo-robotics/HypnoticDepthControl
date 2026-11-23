"""TensorRT 8.x engines of the opset-13 policy step (lethe.deploy.edge.trt), built on the board itself (Jetson, JetPack 4; Python 3.6):
  python3 build_trt.py <dir> [fp32,fp16]        <dir>/policy_op13.onnx -> <dir>/policy_<precision>.plan (kept while newer than the ONNX)
fp32: everything float32. fp16: the net/ region (encoder, readout, mixing, layer norms, head) in float16 under precision constraints, the
features (in/) and the recurrences / infusion (rec/) float32, as lethe.deploy.onnx_export's strongly typed fp16 model. TensorRT may fuse
the whole step into one node and keep it in float32 regardless (TensorRT 8.2 on the Nano does); the build reports the layers' types."""
import collections, json, os, sys
import tensorrt as trt

d = sys.argv[1]
which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["fp32", "fp16"]
src = os.path.join(d, "policy_op13.onnx")
log = trt.Logger(trt.Logger.WARNING)
MOVE = (trt.LayerType.CONSTANT, trt.LayerType.SHUFFLE, trt.LayerType.SLICE, trt.LayerType.GATHER, trt.LayerType.CONCATENATION,
        trt.LayerType.IDENTITY)


def build(prec):
    out = os.path.join(d, "policy_%s.plan" % prec)
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(src):
        print("%s: %s is up to date" % (prec, out))
        return
    b = trt.Builder(log)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    ps = trt.OnnxParser(net, log)
    with open(src, "rb") as f:
        if not ps.parse(f.read()):
            raise SystemExit("parse failed: " + "; ".join(str(ps.get_error(i)) for i in range(ps.num_errors)))
    cfg = b.create_builder_config()
    cfg.max_workspace_size = 1 << 26
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if prec == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(net.num_layers):
            L = net.get_layer(i)
            if L.type in MOVE:
                continue                                          # data movement keeps its inputs' type
            half = L.name.startswith("net/")
            L.precision = trt.float16 if half else trt.float32
            for j in range(L.num_outputs):
                if L.get_output(j).dtype in (trt.float32, trt.float16):
                    L.set_output_type(j, trt.float16 if half else trt.float32)
    elif prec != "fp32":
        raise SystemExit("unknown precision " + prec)
    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        raise SystemExit("build failed: " + prec)
    with open(out, "wb") as f:
        f.write(ser)
    eng = trt.Runtime(log).deserialize_cuda_engine(ser)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    dt = collections.Counter()
    for L in info["Layers"]:
        if isinstance(L, dict):
            for io in L.get("Inputs", []) + L.get("Outputs", []):
                f = io.get("Format/Datatype", "?")
                dt["int8" if "Int8" in f else "half" if "Half" in f else "float" if "Float" in f else f] += 1
    print("%s: %d network layers -> %d engine layers, tensor types %s -> %s (%d bytes)" % (prec, net.num_layers, len(info["Layers"]),
                                                                                          dict(dt), out, ser.nbytes))


for p in which:
    build(p)
