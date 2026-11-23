"""Bit-identical reproduction of the golden outputs of a trained model (tests/golden.py, run in a process of its own: float32 like the
golden run): pytest --golden=DIR --golden-model=PKL"""
import os, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def golden(request, tmp_path_factory):
    d, m = request.config.getoption("--golden"), request.config.getoption("--golden-model")
    if not d or not m:
        pytest.skip("--golden=DIR --golden-model=PKL")
    out = str(tmp_path_factory.mktemp("golden"))
    subprocess.run([sys.executable, os.path.join(HERE, "golden.py"), m, "--out", out], check=True)
    return d, out


@pytest.mark.parametrize("f", ["eval.npz", "grads.npz", "onnx.npz"])
def test_arrays(golden, f):
    d, out = golden
    if not os.path.exists(os.path.join(d, f)):
        pytest.skip(f"no {f}")
    a, b = np.load(os.path.join(d, f)), np.load(os.path.join(out, f))
    assert set(a.files) == set(b.files)
    bad = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)]
    assert not bad, bad[:10]


@pytest.mark.parametrize("f", ["model.bin", "trace.bin"])
def test_cpp_files(golden, f):
    d, out = golden
    assert open(os.path.join(d, f), "rb").read() == open(os.path.join(out, f), "rb").read()
