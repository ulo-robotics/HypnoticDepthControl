"""The baselines: UKF-MM reduces to UKF+LQR; the MPC ones run (CasADi, short episodes)"""
import jax
import numpy as np
import pytest

from lethe.core import model as M
from lethe.baselines import classic as BL


def test_ukfmm_reduces_to_ukflqr(pn):
    _, u0 = M.run_numpy(BL.UKFLQR(q_theta=5e-4, rho=2e-4), pn)
    _, u1 = M.run_numpy(BL.UKFMM(dset=(0,), adapt=False), pn)
    assert np.max(np.abs(u0 - u1)) <= 1e-9


def test_ukfmm_bank_in_range(pn):
    _, u = M.run_numpy(BL.UKFMM(), pn)
    assert np.isfinite(u).all() and (u >= 0).all() and (u <= M.UMAX).all()


@pytest.mark.parametrize("name", ["MPC-MM", "MHE+NMPC"])
def test_mpc_runs(name):
    pytest.importorskip("casadi")
    import lethe.baselines as B
    pn = M.to_numpy(M.sample_patients(jax.random.PRNGKey(9), 2, dict(M.VAL["delay_30s"][0], T=2.0)))
    b, u = M.run_numpy(B.mpc.BASELINES[name](), pn)
    assert b.shape == u.shape == (2, 24)
    assert np.isfinite(u).all() and (u >= 0).all() and (u <= M.UMAX).all()
