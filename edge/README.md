# Lethe and the baselines on edge boards

`edge.py` builds and times the deployed controller (`cpp/lethe_infer.hpp`) and the classical baselines (UKF-MM, MPC-MM, MHE+NMPC, as
CasADi-generated C with IPOPT) on real boards, over ssh. Everything is compiled on the board with its own compiler, and every benchmark first
replays recorded closed loops of the reference implementation (the JAX policy, the Python baselines) and reports the largest difference.

```
python edge.py prepare runs/finetune/best_model.pkl --out runs/finetune/edge [--tensorrt] [--coral --edgetpu-compiler PATH]
python edge.py run rpi5 user@192.168.1.20 local --dir runs/finetune/edge                     # standard library + ssh only
python edge.py run nano --dir runs/finetune/edge --what lethe,ukfmm,tensorrt
python edge.py run rpi4 --dir runs/finetune/edge --what coral --coral-python '~/coral/venv/bin/python'
python edge.py table runs/finetune/edge
```

Hosts are ssh destinations (aliases from `~/.ssh/config`, `user@host`; more options with `--ssh "-p 2222"`), or `local`. Key
authentication is required (`BatchMode`). Results are appended to `<dir>/results/<host>.jsonl`, one JSON line per benchmark with the
board's model, kernel, core, frequency governor and compiler; `table` prints the latest median per-step latency of each.

| benchmark | source | needs on the board |
|---|---|---|
| `lethe` | `cpp/bench.cpp` (float and double) | g++ with C++17 |
| `ukfmm`, `mpcmm` | `edge/baselines/bench_base.cpp` | g++, IPOPT headers (`pkg-config ipopt`; Debian: `coinor-libipopt-dev`) |
| `mhe` | `edge/baselines/bench_mhe.cpp` | the same |
| `tensorrt` | `edge/tensorrt/{build_trt.py,trt_bench.cpp}` | TensorRT 8.x Python bindings, CUDA (JetPack 4) |
| `coral` | `edge/coral/bench.py` | a Coral USB Edge TPU, `libedgetpu`, Python with numpy and `tflite_runtime` |

Each benchmark runs pinned to one core (`taskset`; `--core auto` takes the last). The generated MPC C is large (the dense NMPC Hessian
is about 175 k instructions of straight-line code): building it takes about 6 minutes on a Raspberry Pi 5 and over 20 on a Pi 4 or a
Jetson Nano. Object files stay on the board (`--remote-dir`, default `~/lethe_edge`) and are rebuilt only when their source changes.

The TensorRT path converts the ONNX step to opset 13 with batch 1 (TensorRT 8.2's parser, `lethe/deploy/edge/trt.py`) and builds
fp32 and fp16 engines on the board. On a Jetson Nano (TensorRT 8.2.1) both run as one fused float32 node. int8 is not part of this
path: on the Nano, TensorRT 8.2 reports no native int8 support for the explicit quantize / dequantize model and keeps an implicitly
quantized one in float32.

## Coral Edge TPU

`prepare --coral` (needs TensorFlow) splits the policy: the gated mixing of every S4D layer and the first layer of the head run in int8
on the TPU (four calls per step), everything else (the recurrences, GELUs, layer norms, decoder, physics modes) in float32 on the host. Only
matrix products, additions, products of two tensors and the logistic go to the TPU, with every constant a full-length vector: with these
the TPU's int8 outputs equal TFLite's CPU kernels bit for bit, and the benchmark checks that on every call. Element-wise chains (GELU),
broadcast scalar constants and int8 RSQRT come out differently from the CPU, or not at all, with the compiler available (14.1). `prepare`
reports the int8 policy's closed-loop error against the float one next to the error of reading BIS as an integer.

The compiler runs on x86-64 only. Google's apt repository is gone; the binary is in `github.com/google-coral/edgetpu` under
`compiler/x86_64` (pass `--edgetpu-compiler .../compiler/x86_64/edgetpu_compiler`). On a Raspberry Pi OS bookworm board:

```
sudo dpkg -i libedgetpu1-std_16.0tf2.17.1-1.bookworm_arm64.deb          # github.com/feranick/libedgetpu, built for TensorFlow 2.17.1
python3 -m venv ~/coral/venv                                              # the system Python 3.11 (the wheel is cp311)
~/coral/venv/bin/pip install "numpy<2" tflite_runtime-2.17.1-cp311-cp311-linux_aarch64.whl   # github.com/feranick/TFlite-builds
```

The user needs access to the USB device (the package's udev rules give it to `plugdev`); the stick re-enumerates (`1a6e:089a` ->
`18d1:9302`) when the runtime first loads its firmware.
