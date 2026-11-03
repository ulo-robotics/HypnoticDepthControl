"""
Shared fixtures. The default run is the float64 CPU suite; `pytest -m gpu` runs the float32 GPU engine tests alone (float64 is a global JAX
switch, so they need a process of their own; selftest.py runs both). `--golden=DIR` adds the golden tests against the outputs written by
tests/golden.py for the model given by `--golden-model=PKL` (the = form: pytest reads these before it loads tests/conftest.py).
"""
import jax
import jax.numpy as jnp
import pytest


def pytest_addoption(parser):
    parser.addoption("--golden", default=None, help="folder of the golden outputs (tests/golden.py)")
    parser.addoption("--golden-model", default=None, help="the best_model.pkl they were written from")


def gpu_run(config):
    return config.getoption("-m") == "gpu"


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: float32 GPU engine tests (run alone: pytest -m gpu)")
    if not gpu_run(config):
        jax.config.update("jax_enable_x64", True)


def pytest_collection_modifyitems(config, items):
    if gpu_run(config):
        return
    skip = pytest.mark.skip(reason="GPU engine test: pytest -m gpu")
    for it in items:
        if "gpu" in it.keywords:
            it.add_marker(skip)


def f64(tree):
    return jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if x.dtype == jnp.float32 else x, tree)


def rel(ga, gb):
    """largest relative difference over the leaves of two trees (each leaf against its own scale)"""
    pairs = zip(jax.tree_util.tree_leaves(ga), jax.tree_util.tree_leaves(gb))
    return max(float(jnp.max(jnp.abs(x - y)) / (jnp.max(jnp.abs(y)) + 1e-30)) for x, y in pairs)


@pytest.fixture(scope="session")
def cfg():
    from lethe.core import policy as P
    return dict(P.ARCH_DEFAULTS, H=8, NS=4, L=2, head=8)


@pytest.fixture(scope="session")
def pts():
    """hard patients (delays, target / noise steps, PD / PK shifts, signed multi-step stimuli) with their own episode lengths, float64"""
    from lethe.core import model as M
    return f64(M.sample_patients(jax.random.PRNGKey(3), 32, dict(M.with_stimulus(M.VAL["mixed_hard"][0], "signed"), len_lo=10.0, len_hi=30.0)))


@pytest.fixture(scope="session")
def pn(pts):
    from lethe.core import model as M
    return M.to_numpy(pts)


@pytest.fixture(scope="session")
def p0(cfg):
    """an untrained policy (the prior compensator, zero residual)"""
    from lethe.core import policy as P
    return f64(P.init_params(jax.random.PRNGKey(0), cfg))


@pytest.fixture(scope="session")
def p1(p0):
    """the same physics with a non-trivial residual network"""
    k = jax.random.PRNGKey(1)
    p = jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(k, x.shape, x.dtype), p0)
    p["phys"] = p0["phys"]
    return p


def noisy(cfg, key=None, dtype64=True):
    from lethe.core import policy as P
    key = jax.random.PRNGKey(1) if key is None else key
    p = P.init_params(jax.random.PRNGKey(0), cfg)
    p = f64(p) if dtype64 else p
    return jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(key, x.shape, x.dtype), p)
