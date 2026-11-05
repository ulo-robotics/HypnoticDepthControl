"""Prior design, physics modes, S4D memories and parameter counts of a model.  python paper/model_info.py [MODEL] [OUT]
(default runs/finetune/best_model.pkl -> runs/finetune/paper/model_info.json)"""
import json, os, sys
import numpy as np
from lethe.core import model as M
from lethe.core import policy as P
MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/finetune/best_model.pkl"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(MODEL), "paper", "model_info.json")
m = P.load_model(MODEL)
cfg = m["cfg"]
d = P.design_prior(*(float(cfg[k]) for k in ("q_theta", "sy_scale", "rho_lqr", "q_x")))
tau = lambda lam: sorted(float(-M.DT / np.log(abs(l))) for l in lam)
pp = m["params"]
def cnt(x):
    if isinstance(x, dict):
        return sum(cnt(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(cnt(v) for v in x)
    return int(np.asarray(x).size)
lam = np.exp(-np.exp(np.asarray(pp["phys"]["a"], float)) + 1j * np.asarray(pp["phys"]["b"], float))
init = P.init_params(__import__("jax").random.PRNGKey(0), cfg)
lam0 = np.exp(-np.exp(np.asarray(init["phys"]["a"], float)) + 1j * np.asarray(init["phys"]["b"], float))
mem = []
for ly in pp["extra"]["layers"]:
    dt, A = M.s4d_A(np.asarray(ly["log_dt"]), np.asarray(ly["log_A_re"]), np.asarray(ly["A_im"]))
    mem.append(np.abs(np.exp(np.asarray(dt) * np.asarray(A))).ravel())
mm = -1 / np.log(np.concatenate(mem))
ex = pp["extra"]
out = dict(cfg=cfg, prior_tau_min=[t / 1 for t in tau(d["lam"])], prior_real=bool(np.all(np.abs(np.imag(d["lam"])) < 1e-12)), u0=d["u0"],
           phys_tau_min=tau(lam), phys_tau0_min=tau(lam0), phys_rel_change=float(np.max(np.abs(lam - lam0) / np.abs(1 - lam0))),
           total=cnt(pp), phys=cnt(pp["phys"]), extra=cnt(ex), layer=cnt(ex["layers"][0]), wmix=int(np.asarray(ex["layers"][0]["W_mix"]).size),
           enc=cnt(ex["enc_W"]) + cnt(ex["enc_b"]), head=cnt({k: v for k, v in ex.items() if k.startswith("head")}),
           dec=cnt(ex["dec_W"]) + cnt(ex["dec_b"]), n_phys=int(np.asarray(pp["phys"]["a"]).size),
           s4d_mem_steps=dict(median=float(np.median(mm)), min=float(mm.min()), max=float(mm.max()),
                              at_bound=float(np.mean(mm > 0.99 * M.TAU_MAX / M.DT))))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "cfg"}, default=float))
