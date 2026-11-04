"""edge.py: the C++ engine built and run through the edge runner on this machine (host `local`), and the opset-13 ONNX conversion"""
import json, os, shutil
import jax
import numpy as np
import pytest

from lethe.core import policy as P


@pytest.fixture(scope="module")
def trained_like():
    cfg = dict(P.ARCH_DEFAULTS, H=16, NS=8, L=2, head=16)
    p = P.init_params(jax.random.PRNGKey(0), cfg)
    k = jax.random.PRNGKey(1)
    q = jax.tree_util.tree_map(lambda x: np.asarray(x + 0.05 * jax.random.normal(k, x.shape, x.dtype), np.float32), p)
    q["phys"] = jax.tree_util.tree_map(np.asarray, p["phys"])
    return q


def test_edge_local(tmp_path, trained_like):
    if shutil.which("g++") is None:
        pytest.skip("no g++")
    from lethe.deploy import cpp
    from lethe.deploy.edge import runner as edge
    d = tmp_path / "edge"
    os.makedirs(d / "lethe")
    cpp.write_model(trained_like, str(d / "lethe" / "model.bin"))
    cpp.write_trace(trained_like, str(d / "lethe" / "trace.bin"), 3)
    recs = edge.run_host("local", str(d), ["lethe"], remote=str(tmp_path / "board"), repeats={"lethe": 3}, log=lambda *a: None)
    assert len(recs) == 1 and recs[0]["what"] == "lethe" and recs[0]["board"]["cores"]
    assert max(recs[0][p]["max_abs_err_vs_jax"] for p in ("float", "double")) <= 5e-3
    line = json.loads(open(d / "results" / "local.jsonl").read().splitlines()[-1])
    assert line["host"] == "local" and line["repeats"] == 3


def test_onnx_opset13(tmp_path, trained_like):
    pytest.importorskip("onnxruntime")
    from lethe.deploy import cpp
    from lethe.deploy.edge import trt as onnx_legacy
    cpp.write_trace(trained_like, str(tmp_path / "trace.bin"), 2)
    r = onnx_legacy.export(trained_like, str(tmp_path / "p13.onnx"), str(tmp_path / "trace.bin"))
    assert r["vs_opset18"] <= 1e-5 and r["vs_jax"] <= 5e-3, r
