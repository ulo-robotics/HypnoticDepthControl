"""A population of different configurations trained together == each member trained alone"""
import jax
import numpy as np

from lethe.train import train as T
from lethe.train import trainer as TR
from conftest import rel

CFGS = [dict(H=8, NS=4, L=2, head=8, n_extra=3, opt="adabelief", sched="exp"),
        dict(H=16, NS=8, L=1, head=0, n_extra=0, opt="lion", sched="cosine", warm=25, w_low=3.0),
        dict(H=8, NS=8, L=1, head=16, n_extra=0, opt="sgd", sched="wsd", ema=0.9)]


def test_population_equals_alone():
    cfgs = [dict(T.CONFIG, **o, batch=16, patient_clip=4.0) for o in CFGS]
    up = lambda x: np.asarray(x, np.float64) if np.asarray(x).dtype == np.float32 else x
    st = [jax.tree_util.tree_map(up, TR.new_state(c, jax.random.PRNGKey(i))) for i, c in enumerate(cfgs)]
    key = jax.random.PRNGKey(10_000)
    pop = TR.Population(cfgs, st, key, opt_names=TR.OPT_NAMES, log=lambda m: None, block=3)
    pop.train_to(3)
    together = pop.member_states()
    for i, c in enumerate(cfgs):
        alone = TR.Population([c], [st[i]], key, log=lambda m: None, block=3)
        alone.train_to(3)
        assert rel(together[i]["params"], alone.member_states()[0]["params"]) <= 1e-9, c["opt"]
