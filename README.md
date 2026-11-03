# Lethe

**Closed-loop propofol anaesthesia with a physics-initialized state-space policy, trained end to end through a differentiable patient simulator.**

*Lethe* is the river of forgetting in Greek myth. Here it is a controller that keeps a patient at a target depth of hypnosis: every 5 s it
reads the BIS monitor and sets the propofol infusion. It starts as a classical Kalman observer + LQR designed on a nominal patient, and is then
trained with exact pathwise gradients over a broad, randomized patient population to handle what the classical design does not model
(monitor delay, unknown Hill shape, drifting sensitivity, stimulation, changing targets and noise).

## Problem
- **Plant.** Three-compartment propofol PK (Schnider nominal) + effect site, Hill PD to BIS, 5 s steps, infusion in [0, 100] mg/min.
  Episodes are 30 min (induction + maintenance) unless a scenario says otherwise: the episode length is part of the scenario (`T`).
- **What every patient draws.** PK parameters (lognormal + a common shift), C50, Hill slope / E0 / Emax, Ornstein–Uhlenbeck drift of C50,
  surgical stimulation (steps in log C50 of random size and time), BIS-monitor delay (0–60 s), measurement noise level (with changes), and
  the target BIS (45–60, with steps).
- **Training distribution.** `model.TRAIN`, with signed stimulation: a stimulated patient gets 1 to 3 steps, each raising or lowering C50.
  The fine-tuning trains on episodes of their own length per patient (30–120 min), the cost of each averaged over its own steps.
- **Cost.** J = mean (BIS − target)², on the *true* BIS, plus a penalty on time below BIS 40.
- **Scenarios.** 11 validation scenarios (`model.VAL`: nominal, stimulus, fast drift, 30 / 60 s delay, wide Hill slope, PK shift, target
  range / steps, noise steps, everything mixed) and 11 out-of-distribution stress scenarios (`model.STRESS`, including stimulation that
  lowers C50 and signed multi-step stimulation).

## Method
- **Policy.** `u = r · (u_lin + 10 · residual)`.
  - `u_lin` is the prior compensator: a steady-state Kalman observer on PK + a C50 offset state and an LQR with a moving set point, designed on
    the nominal patient and written as complex modes. Their poles and gains are trainable.
  - The residual is an encoder of three features (the BIS error, the previous infusion, the effect-site estimate from the measured BIS), L
    diagonal S4D layers (H channels, NS complex states each, gated mixing, layer norm) and an MLP head. Its decoder starts at zero, so the
    untrained policy *is* the prior compensator. Every S4D mode's time constant is bounded by 30 min (`model.TAU_MAX`, the shortest
    training episode), so the policy has no notion of elapsed time and holds over episodes of any length.
  - `r = w_ref(target) / C50` puts the physics branch in target-normalized units, so a target change rescales it.
- **Training.** Pathwise gradient of the episode cost through the whole closed loop, over fresh randomized patients every iteration
  (25 % exactly nominal; the delay range ramps up over the first 400 iterations).
  - One gradient per patient, each clipped to 4× the batch median norm before averaging: a few unstable patients would otherwise dominate.
  - Per-group learning rates (physics modes, SSM time scales, the rest) and an EMA of the weights.
- **Selection.** The EMA is validated every 100 iterations on new patients; score = weighted mean over the validation scenarios of
  J + 2 · mean relu(40 − BIS)². The best snapshot is kept.
- **Configuration.** The training configuration (optimizer, learning-rate rule, loss weights, architecture sizes, prior design) comes from a
  large meta-optimization (`search.py`, below). `train.py`'s defaults are that search's winner (`trainer.SELECTED`).
- **Fine-tuning.** `finetune.py` continues the search winner on the training distribution with long, variable-length episodes. It runs
  learning-rate / schedule variants as one population, re-ranks them together with the unchanged winner, and takes the best as the model.

## Repository
| path | content |
|---|---|
| `lethe/core/model.py` | constants, scenarios (specs with their episode length), JAX patient sampler, numpy reference plant, metrics, score |
| `lethe/core/policy.py` | prior design, parameter init, the policy step, the differentiable closed loop (`rollout`), `load_model` |
| `lethe/train/trainer.py` | configuration space, optimizers / learning-rate rules, per-patient clipping, the population trainer (many configurations as one program), validation |
| `lethe/train/train.py`, `search.py`, `finetune.py` | train one configuration; the meta-optimization; the fine-tuning of its winner |
| `lethe/engine/` | the gradient engines: selection (`__init__.py`), the XLA reference (`xla/`), the blocked CUDA engine (`cuda.py`, `kernels/*.cu`) |
| `lethe/baselines/classic.py` | PID, UKF+LQR (with Hill-slope estimation), UKF-MM (delay-aware multiple-model UKF+LQR with adaptive noise) and its tuning, oracle certainty equivalence |
| `lethe/baselines/mpc.py`, `parallel.py` | MPC-MM and MHE+NMPC (CasADi + IPOPT, optional); closed loops in parallel processes |
| `lethe/evaluate.py` | trained policies vs the baselines on the same fresh patients, paired differences, per-patient metrics, figure |
| `lethe/deploy/` | the exports: `cpp.py` (model.bin + reference episodes for `cpp/`), `onnx_export.py` (ONNX, int8, TensorRT, LiteRT, each checked in closed loops), `bench.py` (C++ vs JAX) |
| `cpp/lethe_infer.hpp`, `cpp/bench.cpp` | the deployed controller as one dependency-free C++17 header; its benchmark |
| `embedded/esp32/` | the header as an ESP-IDF app on emulated ESP32s (QEMU), with closed loops against onnxruntime |
| `tests/`, `selftest.py` | the test suite (pytest) and its one-command runner |
| `train.py`, `search.py`, `finetune.py`, `evaluate.py`, `export_cpp.py`, `export_onnx.py`, `bench_infer.py` | command lines (each calls its module's `main`) |

## Use
    pip install -r requirements.txt                                   # GPU: jax[cuda12]; or the Dockerfile. CasADi only for the MPC baselines
    python selftest.py                                                # the tests (CPU, a few minutes; the GPU engine tests too when there is a GPU)
    python train.py --out runs/lethe                                  # 6000 iterations, batch 512; Ctrl-C is safe, then --resume
    python search.py --out runs/search --stimulus signed              # the full meta-optimization (hours, one GPU); --resume
    python finetune.py --search runs/search --out runs/finetune       # 30-120 min episodes (--minutes)
    python evaluate.py runs/finetune/best_model.pkl --stress --prior --plot
    python evaluate.py runs/finetune/best_model.pkl --minutes 250     # every scenario over 250 min
    python evaluate.py runs/finetune/best_model.pkl --baselines UKF-MM,MPC-MM,MHE+NMPC --n 500   # the MPC baselines (parallel processes)
    python bench_infer.py runs/finetune/best_model.pkl                # C++ inference engine: check against JAX, latency
    python export_onnx.py runs/finetune/best_model.pkl --out runs/finetune/onnx --tensorrt fp32,fp16,int8 --litert fp32,dynamic
    python -m lethe.baselines.classic tune                            # UKF-MM's design, on patients of its own (CPU)

Docker (GPU): `docker build -t lethe .` and
`docker run --rm --gpus all -v $PWD/runs:/app/runs lethe python train.py --out runs/lethe`.

Every training setting is a flag of `train.py`. A run writes `config.json`, `log.jsonl` (one line per validation), `checkpoint.pkl` (for
`--resume`) and `best_model.pkl` (the best EMA weights). The engine choice is automatic (`--engine raw|adjoint|cblk` forces one,
`--full-tune` tunes every candidate); compiled kernels and engine decisions are cached in `~/.cache/lethe` (`LETHE_CACHE`).

## Training engines
Per-patient gradients through a closed loop of hundreds of steps are the whole cost of training. `lethe/engine` computes them for a
population of networks (one network in `train.py`) in one of two ways:
- **Raw JAX:** `vmap(grad)` through `policy.rollout`. Always available; the reference, and the fallback.
- **Adjoint:** exact up to float32 rounding.
  - A forward sweep writes a tape (the small carry and the S4D outputs per step); a backward sweep propagates only the carry and S4D adjoints.
  - The XLA reference (`xla/`, plain JAX on pytrees: complex states, every function at module level and jitted) runs both sweeps as scans
    (per step, or with the S4D blocked over T steps: an inner scan per block, a Toeplitz contraction for the inputs within the block) and
    turns the tapes into every per-patient parameter gradient: per patient one VJP of the replayed step over all its steps, plus an FFT
    cross-correlation for the S4D kernels. The backends exchange tapes in the CUDA kernels' format, so their sweeps combine freely.
  - The blocked CUDA engine (`cuda.py`) keeps a thread block of 8–32 patients: the GLU and head matmuls run on tensor cores across them
    (3xTF32: float32 accuracy up to rounding), and the S4D state stays in global memory, touched once per T-step block. Its backward also
    writes the per-step cotangents, and `kernels/casm.cu` assembles the whole gradient from them in one pass, without replaying the steps.

**How the engine is chosen.** For each shape (sizes, population size, batch, episode length), the candidates are compiled, checked against
the XLA reference on stable patients, and timed on the real batch. A shape is tuned once: another population size of it, or a decision
recorded by an earlier version of the kernels, only re-checks the choice made there. A new shape first tries the candidates that won at every
measured shape and tunes every candidate only if none of them works. Anything that fails to compile or disagrees with the reference is never
used. Raw JAX is used when no adjoint candidate works and for runs too short to repay tuning.

Per-patient gradient of a population of 8 networks of the selected configuration (H32 NS16 L3 head64), 512 patients of 30 min, warm GPU
(RTX 4000 Ada), median: forward sweep 67 ms (XLA) → 33 ms (CUDA), backward sweep 110 ms → 65 ms, whole gradient 288 ms (XLA sweeps and
assembly) → 110 ms (CUDA forward, backward with tapes and tape assembly).

The blocked CUDA kernels at the search shapes (population of 8, 512 patients, warm GPU, the engine's choice from an empty cache, median of
11 calls). The kernels pipeline the block-end loads of the S4D modes, read the tensor-core weights in fragment order as plain float32 and
split every operand into its TF32 hi / lo parts in registers (hi = the top 19 bits, a mask; lo = x − hi, exact in float32), compute the
sigmoid and GELU on the special-function unit, take the two reductions of the layer-norm backward in one pass, and are built for a fixed
batch size:

| shape (L H NS) | whole gradient |
|---|---|
| 3 64 32 | 225 ms |
| 2 64 32 | 145 ms |
| 3 32 32 | 112 ms |
| 3 32 16 | 92 ms |
| 3 16 8 | 40 ms |

Tapes scale with members × batch × steps: long episodes need smaller populations (the fine-tuning trains 4 variants at 120 min).

## Inference
`cpp/lethe_infer.hpp` is the deployed controller as a single dependency-free C++17 header: `Model::load` the file written by
`export_cpp.py` (or `Model::parse` it from memory), then call `step(measured BIS, target, previous target)` every 5 s, `reset()` at the
start of an episode. `bench_infer.py` replays closed-loop episodes recorded with the JAX policy and checks the C++ output step by step,
then measures latency. For the selected configuration (H32 NS16 L3 head64) on an Intel Core Ultra 9 285: 4.6 µs per step in float
(5.0 in double), agreeing with the JAX policy to 1.5e-3 mg/min (float32 rounding); JAX itself, jitted for one patient, needs 340 µs.

`export_onnx.py` writes the same step, batched over patients, as an ONNX graph (opset 18, float32; the recurrent states are inputs and
outputs), and from it reduced-precision models: `policy_int8.onnx` (quantize / dequantize nodes), TensorRT engines (`--tensorrt
fp32,fp16,int8`) and LiteRT models at batch 1 (`--litert fp32,dynamic`). Each is checked in its own runtime, in closed loops on the numpy
plant over every validation scenario, against the float32 model.

Precision has to be judged clinically here. The infusion is a high-gain function of the measured BIS (du/dBIS 3.6 mg/min per point at the
median, up to 40), and a rounding anywhere in the network acts like a small change of that measurement. So the check reports the selection
score, the infusions and the BIS trajectories, beside a physical floor: BIS read as an integer, as monitors display it. For the model of
the paper, 256 patients × 11 scenarios against onnxruntime float32:

| model | score change | \|du\| p99 (mg/min) | \|dBIS\| p99 | \|dBIS\| max |
|---|---|---|---|---|
| floor: BIS read as an integer | +0.070 | 4.7 | 0.50 | 6.6 |
| TensorRT fp32, LiteRT fp32 | < 1e-6 | 8e-5 | 0 | 0 |
| TensorRT fp16 (network fp16, recurrences fp32) | +0.001 | 0.17 | 0.025 | 0.20 |
| LiteRT dynamic (int8 weights) | +0.011 | 0.87 | 0.11 | 0.90 |
| int8, onnxruntime and TensorRT (identical) | +0.035 | 2.0 | 0.30 | 8.8 |

The int8 model quantizes the mixing and head matmuls after smoothing the activations per channel (as in SmoothQuant); the encoder, the
recurrences and the infusion stay float32. Recurrent states in integers do not hold up (an int8 S4D state breaks the controller).

**Microcontrollers.** `embedded/esp32` runs the header on emulated ESP32, ESP32-S3 and ESP32-C3 chips (QEMU). A control step takes about
160 k instructions on the ESP32 / S3 (about 0.7–1.1 ms at 240 MHz, from the cycles per instruction of ESP-DSP's published kernels) and
4.8 M on the C3 (software floating point, 30–48 ms). In closed loops against onnxruntime, float on the ESP32 / S3 and double on the S3
change nothing clinically (p99 |du| 7e-5 / 1.3e-4 mg/min).

## Meta-optimization
`search.py` follows the procedure used for the paper:
1. **Prior screen.** Score 4096 prior designs at iteration 0 and keep the best 25 % as the pool for the trials' prior design.
2. **Trials.** 1025 trials: trial 0 is the base configuration, the rest a scrambled Sobol sequence over `trainer.SPACE`.
3. **Successive halving.** Rungs at 300 / 900 / 2700 iterations, η = 3. A trial is ranked by its best snapshot: its EMA is validated every
   100 iterations on new patients (and at the end of each rung on the rung's own key), and the best EMA is kept.
4. **Final stage.** The top 3 configurations continue for 3000 iterations with 3 training-data seeds each.
5. **Re-rank and test.** The best snapshots of all final models are re-ranked on a fresh key with 4096 patients per scenario; the configuration with the best mean
   over seeds wins. The winner is scored once on the test key against the untrained base prior.
6. **Fine-tuning** (`finetune.py`). The winner continues for 3000 iterations from its EMA weights with a fresh optimizer state, as one
   population of learning-rate × schedule variants. Each variant's best validated snapshot and the unchanged winner are re-ranked on a fresh
   key. The best is tested once on the test key, against the winner and the untrained base prior.

All trials see the same training patients. Trials train in **populations** (`trainer.Population`): the trials of one network shape, up to 8 at a
time, as one jitted program, each with its own hyperparameters and optimizer:
- the members are zero-padded to a common shape, and masks keep each one exactly its own network: padded entries get exactly zero gradients,
  so a member trains like the same configuration alone (tested);
- every hyperparameter is an array, the optimizer is a switch over the members' update rules, and 100 iterations run per jitted call, so one
  compilation serves every population of that shape, in every rung;
- many recurrences at once keep the GPU busy, where one trial's 512 patients leave most of it idle.

One process owns the GPU. A population that does not fit in memory is split in two. The search is resumable at population granularity, and
every score is logged.

## Baselines
`lethe/baselines` holds the classical references, all on the numpy plant: PID, UKF+LQR (optionally estimating the Hill slope), UKF-MM (a bank
of delay hypotheses, each a UKF on PK + C50 with its own Smith predictor, probability-weighted LQR laws, adaptive measurement noise; the
strongest estimator-based design), and the oracle certainty-equivalence controller (it knows the patient: a bound, not a controller). The MPC
baselines need CasADi:
- **MPC-MM:** UKF-MM's estimator (`UKFMM.estimate`) with a robust nonlinear MPC in place of its LQR mixture: the probability-weighted score
  cost over the observers' predictions, 5 min horizon in move blocks, IPOPT.
- **MHE+NMPC:** full-state moving-horizon estimation per delay hypothesis (the PK states and a random-walk C50, with the Hill slope, the
  effect-site rate and E0 as parameters, process noise and arrival priors), then the same robust NMPC.

CasADi's thread map serializes IPOPT, so `evaluate.py` runs them in spawned processes (`--workers`).

## Reproducibility
- **Random streams.** Training patients from key `10000 (+ seed)` (`20000` in the fine-tuning), validation from `12345` folded with the
  iteration (rung / stage in the search, 50000 + iteration in the fine-tuning), re-rank `424242` (folded with 1 in the fine-tuning), test
  `99999`, evaluation `8000000`, UKF-MM tuning `7000000`. No set of patients is used for two purposes.
- **Determinism.** Every run is deterministic given its configuration and seed, up to the floating-point order of the chosen engine.
- **Golden outputs.** `tests/golden.py` writes the outputs of a trained model (per-patient evaluation metrics of the policy and the
  baselines, raw per-patient gradients, the C++ export, the ONNX graph's outputs); `pytest --golden=DIR --golden-model=PKL` checks that the
  code reproduces them bit for bit.

## Verification (`python selftest.py`)
- **Float64, CPU:** the untrained policy equals an independent numpy implementation of the prior compensator; the training closed loop equals
  the numpy reference plant step by step; the pathwise gradient matches finite differences; the mean of per-patient gradients equals the
  batch gradient; episode lengths are data (a patient's own length only weights its cost); a saved model with the earlier 4-input encoder
  converts to the same policy; the adjoint gradients (per-step and blocked XLA sweeps) equal raw autodiff per patient, also through the
  engine interface (the CUDA kernels' tape format); padding is exact (zero gradients on the padding); a population equals its members trained alone; UKF-MM with one observer
  of delay 0 equals UKF+LQR; the MPC baselines run; the C++ engine and the ONNX graph agree with the JAX policy in closed loop.
- **Float32, GPU** (`pytest -m gpu`, a process of its own): every CUDA sweep and assembly candidate equals the XLA references, per tape and
  per patient and parameter, for three single networks and a padded population.
- **Lint:** `ruff check .`

## Results (the paper's run)
The paper's run, its models (an encoder with a time input, unbounded S4D time constants) and the numbers below belong to the code at tag
`pre-refactor`; this code does not load those models. `search.py` (1025 trials, stimulus at a fixed time) → winner config 171: nadamw, wsd,
lr 7.4e-3, H 32, NS 16, L 3, head 64, 8 extra physics modes. `finetune.py` on the random stimulus → test score 77.75; the winner without
fine-tuning 78.77, the untrained base prior 134.12.

`evaluate.py` (2000 fresh patients per scenario, the same patients for every controller; score = weighted J + 2 viol):

| controller | score |
|---|---|
| **policy (fine-tuned)** | **77.96** |
| search winner (not fine-tuned) | 78.94 |
| UKF-MM | 83.03 |
| MPC-MM | 83.46 |
| UKF+LQR+γ | 132.56 |
| UKF+LQR | 161.66 |
| PID | 245.64 |
| untrained policy (prior compensator) | 110.43 |
| oracle CE (knows the patient; a bound, not a controller) | 67.41 |

MHE+NMPC, on the first 500 of those patients: 83.23 (policy 78.24, UKF-MM 83.71 there). The policy beats it on every scenario.

- **Against UKF-MM** (policy − UKF-MM, J, 95 % CI): from −0.2 to −3.6 on the single-factor scenarios (gamma_wide +0.07 ± 0.27, within noise),
  −13.9 ± 1.7 on mixed_hard and −35.5 ± 3.3 on everything_ood. Worse only at high measurement noise: noise_8 +4.0, noise_bursts +3.5.
- **Long episodes.** That model's maintenance error grew past 30 min (it had seen only 30-min episodes, and a time input). Fine-tuned
  from it without the time input, with bounded time constants and on 30–120 min episodes, the policy holds a flat maintenance cost over
  250 min (per 30-min window, nominal 3.3–3.9 against UKF-MM's 3.6–3.8; mixed_hard 15–17 against 24–28).

## Status
- **Done and verified:** plant, policy, baselines (classic and MPC), training (one configuration or a population), evaluation, the training
  engines, the meta-optimization, the fine-tuning, the C++ inference engine, the ONNX / TensorRT / LiteRT exports, the ESP32 app.
- **Running:** a new search and fine-tuning with this setup (signed multi-step stimulation, variable-length episodes, no time input).
