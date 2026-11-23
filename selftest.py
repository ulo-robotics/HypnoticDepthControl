#!/usr/bin/env python
"""
The test suite in one command: the float64 CPU tests (tests/, a few minutes), then, when a GPU is present, the float32 GPU engine tests in a
process of their own (pytest -m gpu). Extra arguments go to both pytest runs (e.g. -x, -k adjoint, --golden=DIR --golden-model=PKL).

  float64, CPU: the untrained policy is the prior compensator; the training closed loop equals the numpy plant; gradients against finite
  differences and the batch gradient; episode lengths as data; the 4-row model conversion; the adjoint engine (XLA sweeps, blocked sweeps)
  equals raw autodiff per patient; padding is exact; a population equals its members trained alone; UKF-MM reduces to UKF+LQR; the MPC
  baselines run (with CasADi); the C++ engine and the ONNX graph against JAX (with g++ / onnxruntime).
  float32, GPU: every CUDA sweep and assembly candidate equals the XLA references, for single networks and a padded population.
"""
import subprocess, sys


def main():
    args = sys.argv[1:]
    rc = subprocess.run([sys.executable, "-m", "pytest", "-q", *args]).returncode
    from lethe import engine as E
    if E._gpu():
        rc = rc or subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "gpu", *args]).returncode
    else:
        print("no GPU: the CUDA engine tests are skipped")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
