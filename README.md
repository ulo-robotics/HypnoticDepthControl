# Lethe

**Closed-loop propofol anaesthesia with a physics-initialized state-space policy, trained end to end through a differentiable patient simulator.**

*Lethe* is the river of forgetting in Greek myth. Here it is a controller that keeps a patient at a target depth of hypnosis: every 5 s it
reads the BIS monitor and sets the propofol infusion. It starts as a classical Kalman observer + LQR designed on a nominal patient, and is then
trained with exact pathwise gradients over a broad, randomized patient population to handle what the classical design does not model
(monitor delay, unknown Hill shape, drifting sensitivity, changing targets and noise).

## Problem
- **Plant.** Three-compartment propofol PK (Schnider nominal) + effect site, Hill PD to BIS, 30 min (induction + maintenance), 5 s steps, infusion
  in [0, 100] mg/min.
- **What every patient draws.** PK parameters (lognormal + a common shift), C50, Hill slope / E0 / Emax, Ornstein–Uhlenbeck drift of C50, a
  surgical stimulus, BIS-monitor delay (0–60 s), measurement noise level (with changes), and the target BIS (45–60, with steps).
- **Cost.** J = mean (BIS − target)², on the *true* BIS, plus a penalty on time below BIS 40.
- **Scenarios.** 11 validation scenarios (`model.VAL`: nominal, stimulus, fast drift, 30 / 60 s delay, wide Hill slope, PK shift, target range /
  steps, noise steps, everything mixed) and 9 out-of-distribution stress scenarios (`model.STRESS`).

## Method
- **Policy.** `u = r · (u_lin + 10 · residual)`.
  - `u_lin` is the prior compensator: a steady-state Kalman observer on PK + a C50 offset state and an LQR with a moving set point, designed on
    the nominal patient and written as 5 complex modes. Their poles and gains are trainable.
  - The residual is an encoder (4 features), L diagonal S4D layers (H channels, NS complex states each, gated mixing, layer norm) and an MLP head.
    Its decoder starts at zero, so the untrained policy *is* the prior compensator.
  - `r = w_ref(target) / C50` puts the physics branch in target-normalized units, so a target change rescales it.
- **Training.** Pathwise gradient of the episode cost through the whole 360-step closed loop, over fresh randomized patients every iteration
  (25 % exactly nominal; the delay range ramps up over the first 400 iterations).
  - One gradient per patient, each clipped to 4× the batch median norm before averaging: a few unstable patients would otherwise dominate.
  - Per-group learning rates (physics modes, SSM time scales, the rest) and an EMA of the weights.
- **Selection.** The EMA is validated every 100 iterations on new patients; score = weighted mean over the validation scenarios of
  J + 2 · mean relu(40 − BIS)². The best snapshot is kept.
- **Configuration.** The training configuration (optimizer, learning-rate rule, loss weights, architecture sizes, prior design) comes from a
  large meta-optimization (`search.py`, below). `train.py`'s defaults are that search's winner (`trainer.SELECTED`).

## Repository
| file | content |
|---|---|
| `model.py` | constants, scenarios, JAX patient sampler, numpy reference plant, metrics, score |
| `policy.py` | prior design, parameter init, the policy step, the differentiable closed loop (`rollout`) |
| `baselines.py` | PID, UKF+LQR, UKF+LQR with Hill-slope estimation, oracle certainty equivalence (numpy, vectorized) |
| `trainer.py` | configuration space, optimizers / learning-rate rules, per-patient clipping, the population trainer (many configurations as one program), validation |
| `train.py` | train one configuration (a population of one) |
| `search.py` | meta-optimization: prior screen, Sobol trials in populations, successive halving, multi-seed final stage, re-rank, test |
| `evaluate.py` | trained policy vs the baselines on the same fresh patients, paired differences, per-patient metrics, figure |
| `engine.py` | choice of the gradient engine (below) |
| `kstep.py` | the step in kernel form, the XLA reference sweeps, the shared gradient assembly |
| `kpallas.py` | Pallas (Triton) sweeps |
| `kcuda.py`, `cuda/sweep.cu` | CUDA sweeps, called through XLA FFI |
| `export_cpp.py`, `cpp/lethe_infer.hpp` | C++ inference engine of the deployed controller and its model exporter |
| `bench_infer.py`, `cpp/bench.cpp` | C++ vs JAX inference: step-by-step agreement and latency |
| `selftest.py` | correctness checks |

## Use
    pip install -r requirements.txt                                   # GPU: jax[cuda12]; or the Dockerfile
    python selftest.py
    python train.py --out runs/lethe                                  # 6000 iterations, batch 512; Ctrl-C is safe, then --resume
    python evaluate.py runs/lethe/best_model.pkl --stress --prior --plot
    python bench_infer.py runs/lethe/best_model.pkl                   # C++ inference engine: check against JAX, latency
    python search.py --out runs/search                                # the full meta-optimization (hours, one GPU)
    python search.py --out runs/search --resume

Docker (GPU): `docker build -t lethe .` and
`docker run --rm --gpus all -v $PWD/runs:/app/runs lethe python train.py --out runs/lethe`.

Every training setting is a flag of `train.py`. A run writes `config.json`, `log.jsonl` (one line per validation), `checkpoint.pkl` (for
`--resume`) and `best_model.pkl` (the best EMA weights).

## Training engines
Per-patient gradients through a 360-step closed loop are the whole cost of training. `engine.py` computes them for a population of networks
(one network in `train.py`) in one of two ways:
- **Raw JAX:** `vmap(grad)` through `policy.rollout`. Always available; the reference, and the fallback.
- **Adjoint:** `kstep.py`, exact up to float32 rounding.
  - A forward sweep writes a tape (the small carry and the S4D outputs per step).
  - A backward sweep propagates only the carry and S4D adjoints.
  - An XLA assembly turns both tapes into every per-patient parameter gradient at once: one batched VJP over all (step, patient) pairs, plus an
    FFT cross-correlation for the S4D kernels.
  - The two sweeps run on XLA scans (per step, or with the S4D blocked over T steps: one GEMM gives the state part of a block's outputs
    and the state is read and written once per block, which matters when the state is large), Pallas kernels or CUDA kernels.

**How the engine is chosen.** For each shape (sizes, population size, batch), every sweep candidate is compiled, checked against the XLA
reference on stable patients, and timed on the real batch. The forward and backward sweeps are chosen independently; anything that fails to
compile or disagrees is never used. The Pallas and XLA sweeps run vmapped over a population; the CUDA sweeps serve a single unpadded network.
Raw JAX is used when no adjoint candidate works (it was slower than the adjoint path at every measured shape, and vmapped over a population it
mostly does not fit in memory) and for runs too short to repay tuning. Decisions are cached in `~/.cache/pkpd_v1/engine.json`.
`PKPD_ENGINE=raw|adjoint|pallas|cuda` forces a choice. All engines use exact float32 matmuls (no TF32), so they compute the same function.

Training speed of one configuration (`train.py`), batch 512, on an RTX 4000 Ada (milliseconds per iteration, including sampling,
clipping and the optimizer step; measured before population training, with the same engines):

| size (H NS L head) | raw JAX | selected engine | speed-up | chosen sweeps (forward, backward) |
|---|---|---|---|---|
| 16 8 1 0 | 107 | 14 | 7.6× | Pallas, Pallas |
| 32 8 3 64 | 294 | 49 | 6.0× | CUDA, CUDA |
| 64 8 3 64 (the selected size) | 514 | 108 | 4.8× | CUDA, CUDA |

At the selected size, 6000 iterations take about 11 minutes. Tuning a new shape takes about 1.5 minutes, once.

In the search, populations of 8 trials train at, per trial-iteration (everything included, measured in the full search): L3 H64 NS32
(the largest shape of the space) 99-109 ms, L2 H64 NS32 75 ms; the paper's run needed 165 and 113 ms for these shapes.

## Inference engine
`cpp/lethe_infer.hpp` is the deployed controller as a single dependency-free C++17 header: load the file written by `export_cpp.py`, then
call `step(k, measured BIS, target, previous target)` every 5 s. `bench_infer.py` replays closed-loop episodes recorded with the JAX policy
and checks the C++ output step by step, then measures latency.

For the selected policy (H 64, NS 8, L 3, head 64) on an Intel Core Ultra 9 285:
- **C++, float:** 8.2 µs per control step (p99 8.7 µs); agrees with the JAX policy to 1.8e-3 mg/min (float32 rounding).
- **C++, double:** 9.2 µs per step.
- **JAX, for comparison** (one patient, jitted, dominated by dispatch): 280 µs per step on CPU, 3 ms on GPU.

## Meta-optimization
`search.py` follows the procedure used for the paper:
1. **Prior screen.** Score 4096 prior designs at iteration 0 and keep the best 25 % as the pool for the trials' prior design.
2. **Trials.** 1025 trials: trial 0 is the base configuration, the rest a scrambled Sobol sequence over `trainer.SPACE`.
3. **Successive halving.** Rungs at 300 / 900 / 2700 iterations, η = 3, each rung scored on its own fresh validation key.
4. **Final stage.** The top 3 configurations continue for 3000 iterations with 3 training-data seeds each.
5. **Re-rank and test.** All final models are re-ranked on a fresh key with 4096 patients per scenario; the configuration with the best mean
   over seeds wins. The winner is scored once on the test key against the untrained base prior.

All trials see the same training patients. Trials train in **populations** (`trainer.Population`): the trials of one network shape, up to 8 at a
time, as one jitted program, each with its own hyperparameters and optimizer:
- the members are zero-padded to a common shape (the head and the extra physics modes to the largest of the space), and masks keep each one
  exactly its own network: padded entries get exactly zero gradients, so a member trains like the same configuration alone (`selftest.py`);
- every hyperparameter is an array, the optimizer is a switch over the members' update rules, and 100 iterations run per jitted call, so one
  compilation serves every population of that shape, in every rung;
- many recurrences at once keep the GPU busy, where one trial's 512 patients leave most of it idle.

One process owns the GPU (the JAX model: a device belongs to one process). A population that does not fit in memory is split in two. The search
is resumable at population granularity, and every score is logged.

## Reproducibility
- **Random streams.** Training patients from key `10000 (+ seed)`, validation from `12345` folded with the iteration (rung / stage in the search),
  re-rank `424242`, test `99999`, evaluation `8000000`. No set of patients is used for two purposes.
- **Determinism.** Every run is deterministic given its configuration and seed, up to the floating-point order of the chosen engine.

## Verification
- **`selftest.py`** (float64):
  - the untrained policy equals an independent numpy implementation of the prior compensator;
  - the training closed loop equals the numpy reference plant driven step by step (to 4e-13);
  - the pathwise gradient matches finite differences (7e-8);
  - the mean of per-patient gradients equals the batch gradient.
- **Adjoint engine.** Per-patient gradients equal raw autodiff to 1e-14 in float64. The Pallas and CUDA sweeps equal the XLA reference sweeps
  to float32 rounding (≤ 2e-5 relative) on every tape, including no-head, H < 32 and extra-physics-mode shapes.
- **GPU engines.** `selftest.py` also checks every Pallas and CUDA sweep candidate against the XLA reference on the GPU, at three shapes
  including the selected one.
- **Search.** A smoke search (`search.py --smoke`) runs end to end. Interrupted and resumed, it gives the identical winner and test score.
- **Inference.** The C++ engine reproduces the JAX policy's actions to float32 rounding.
- **Against the earlier code base.** Plant and baselines give identical BIS traces (1e-13); the prior design is bit-identical.

## Status
- **Done and verified:** plant, policy, baselines, training (one configuration or a population), evaluation, the training engines, the
  meta-optimization (smoke-tested end to end: the same winner and test score as training the trials one by one), the C++ inference engine.
- **Running:** the full-size meta-optimization (1025 trials) with this code.
- **Differences from the paper's search run** (`fast_test`), to decide: the training setup here uses a nominal-patient share of 0.25 (the run
  used 0.15) and no band term in the loss (the run used `--band-w 1 --band-half 5`). The search space, budgets, population size and keys
  are the run's.
