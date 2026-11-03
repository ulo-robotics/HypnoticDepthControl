"""closed loops of numpy-side controllers in parallel processes: patients are independent, so the batch is split across workers, each with its
own controller (CasADi's thread map serializes IPOPT; separate processes do not share it). Workers are spawned (not forked: the parent may
hold JAX / CUDA state), so the controller factory must be picklable (a class, or functools.partial of one)."""
import multiprocessing as mp
import os
import numpy as np

from lethe.core import model as M


def _work(args):
    make, pts = args
    return M.run_numpy(make(), pts)


def run_parallel(make, pts, workers=None):
    """make() -> controller; pts: a numpy patient batch. Returns (BIS (B, N), u (B, N)) in patient order"""
    B = pts["mu"].shape[0]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    idx = [i for i in np.array_split(np.arange(B), min(workers, B)) if len(i)]
    with mp.get_context("spawn").Pool(len(idx)) as pool:
        res = pool.map(_work, [(make, M.take(pts, i)) for i in idx])
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])
