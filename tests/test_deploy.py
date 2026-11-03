"""The exports: the C++ engine and the ONNX graph against the JAX policy in closed loop"""
import json, os, shutil, subprocess
import jax
import numpy as np
import pytest

from lethe.core import model as M
from lethe.core import policy as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def trained_like():
    """compact float32 parameters with a non-trivial residual (stands in for a trained model)"""
    cfg = dict(P.ARCH_DEFAULTS, H=16, NS=8, L=2, head=16)
    p = P.init_params(jax.random.PRNGKey(0), cfg)
    k = jax.random.PRNGKey(1)
    q = jax.tree_util.tree_map(lambda x: np.asarray(x + 0.05 * jax.random.normal(k, x.shape, x.dtype), np.float32), p)
    q["phys"] = jax.tree_util.tree_map(np.asarray, p["phys"])
    return q


def test_cpp_engine(tmp_path, trained_like):
    if shutil.which("g++") is None:
        pytest.skip("no g++")
    from lethe.deploy import cpp as X
    X.write_model(trained_like, str(tmp_path / "model.bin"))
    X.write_trace(trained_like, str(tmp_path / "trace.bin"), 4)
    assert open(tmp_path / "model.bin", "rb").read(8) == b"LETHE03\0"
    exe = str(tmp_path / "bench")
    subprocess.run(["g++", "-O2", "-std=c++17", "-o", exe, os.path.join(ROOT, "cpp", "bench.cpp")], check=True)
    res = json.loads(subprocess.run([exe, str(tmp_path / "model.bin"), str(tmp_path / "trace.bin"), "5"], check=True, capture_output=True,
                                    text=True).stdout)
    errs = [v["max_abs_err_vs_jax"] for v in res.values() if isinstance(v, dict)]
    assert errs and max(errs) <= 5e-3, res                               # mg/min: float32 rounding over whole episodes


def test_onnx_graph(trained_like):
    pytest.importorskip("onnxruntime")
    from lethe.deploy import onnx_export as EO
    m, d = EO.build(trained_like)
    pts = M.to_numpy(M.sample_patients(jax.random.PRNGKey(2), 4, M.VAL["mixed_hard"][0]))
    assert EO.against_jax(EO.ort_runner(m), trained_like, d, pts) <= 1e-2
