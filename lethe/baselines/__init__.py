"""The baseline controllers: classic.py (PID, UKF+LQR, UKF-MM, the oracle) and mpc.py (MPC-MM, MHE+NMPC; CasADi). run() is the closed loop
of any of them on a numpy patient batch; the MPC ones run in parallel processes."""
from lethe.baselines import classic, mpc

BASELINES = {**classic.BASELINES, **mpc.BASELINES}


def run(name, pts, workers=None):
    """closed loop of baseline `name` on the numpy patient batch pts -> (BIS (B, N), u (B, N))"""
    if name in mpc.BASELINES:
        from lethe.baselines import parallel
        return parallel.run_parallel(mpc.BASELINES[name], pts, workers)
    from lethe.core import model as M
    return M.run_numpy(classic.BASELINES[name](), pts)
