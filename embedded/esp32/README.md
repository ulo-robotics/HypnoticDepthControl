# Lethe on an ESP32

The policy header (`cpp/lethe_infer.hpp`) as an ESP-IDF app, run on emulated chips (QEMU; esp32, esp32s3, esp32c3). The app embeds the
exported model (`Model::parse` reads it from flash) and talks over UART0:

- `B`: bench. Counter calibration, the recorded JAX episodes replayed (accuracy, cycle counts per step), ESP-DSP's plain C kernels
  (published real-chip cycles, to calibrate cycles per instruction) and the libm calls of a step. Prints `LETHE ...` lines, then `LETHE DONE`.
- `F` / `D`: closed loop with a float / double policy. Packets `S` int32 k, float y, tgt, tgt_prev -> float u (k = 0 resets the policy);
  any other byte ends it.

From the repository root (ESP-IDF image `espressif/idf:v5.5.1`; the whole repository is mounted so the app finds `cpp/`):

```
python export_cpp.py runs/lethe/best_model.pkl --out embedded/esp32/main            # model.bin, trace.bin (embedded in the image)
docker run --rm -v $PWD:/repo -w /repo/embedded/esp32 espressif/idf:v5.5.1 bash run_qemu.sh esp32 build
docker run --rm -v $PWD:/repo -w /repo/embedded/esp32 espressif/idf:v5.5.1 bash bench.sh esp32 "-icount shift=0" 5701
```

Closed loops against onnxruntime (`hil.py`, one patient at a time on the numpy plant): start QEMU instances with
`qemus.sh <target> <first port> <instances>` on a docker network, then from the Lethe image
`python embedded/esp32/hil.py model.pkl model.bin <patients> <host>:<port>:F|D ...`. Without devices it compares the C++ engine on the
host (float and double) with onnxruntime.

The serial port must use `nodelay=on` (QEMU), or Nagle's algorithm adds about 44 ms to every reply.
