// The Lethe policy (cpp/lethe_infer.hpp, Model::parse on the embedded model.bin) on an ESP32 in QEMU, driven over UART0 by a host. Commands (one byte):
//   'B'  bench: counter calibration (a block of nops), the recorded JAX episodes replayed (accuracy, counts per step), ESP-DSP's
//        plain C kernels (published real-chip cycle counts) and the libm calls of a step; prints LETHE lines and LETHE DONE
//   'F' / 'D'  closed loop with a float / double policy: packets 'S' int32 k, float y, tgt, tgt_prev -> float u (k = 0 resets);
//        'Q' ends it
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <vector>
#include "driver/uart.h"
#include "esp_chip_info.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "lethe_infer.hpp"

extern "C" {
int dsps_dotprod_f32_ansi(const float* src1, const float* src2, float* dest, int len);
int dspm_mult_f32_ansi(const float* A, const float* B, float* C, int m, int n, int k);
int dsps_biquad_f32_ansi(const float* input, float* output, int len, float* coef, float* w);
}

extern const uint8_t model_bin[] asm("_binary_model_bin_start");
extern const uint8_t model_end[] asm("_binary_model_bin_end");
extern const uint8_t trace_bin[] asm("_binary_trace_bin_start");

static inline uint32_t cc() { return esp_cpu_get_cycle_count(); }

// trace.bin: int32 n, N, then per patient float64 y[N], target[N], u_ref[N]
static double trace_at(int i, int which, int k) {
  int32_t d[2];
  std::memcpy(d, trace_bin, sizeof d);
  double v;
  std::memcpy(&v, trace_bin + 8 + (((size_t)i * 3 + which) * d[1] + k) * sizeof(double), sizeof v);
  return v;
}

template <typename F>
static uint32_t best_of(int reps, F f) {
  uint32_t b = ~0u;
  for (int r = 0; r < reps; r++) {
    const uint32_t c0 = cc();
    f();
    b = std::min(b, cc() - c0);
  }
  return b;
}

static volatile float sink;

static void calibrate() {
  const uint32_t e = best_of(50, [] {});
  const uint32_t n = best_of(50, [] { asm volatile(".rept 4000\n nop\n .endr"); });
  std::printf("LETHE calib: empty %u, 4000 nops %u\n", (unsigned)e, (unsigned)n);
}

static void kernels() {
  static float a[256], b[256], c[256], A[256], Bm[256], Cm[256];
  for (int i = 0; i < 256; i++) { a[i] = std::sin(0.1f * i); b[i] = std::cos(0.07f * i); A[i] = 0.01f * (i % 17); Bm[i] = 0.02f * (i % 13); }
  float dest, coef[5] = {0.1f, 0.2f, 0.1f, -0.5f, 0.2f}, w[2] = {0, 0};
  std::printf("LETHE kernel dotprod256 %u\n", (unsigned)best_of(20, [&] { dsps_dotprod_f32_ansi(a, b, &dest, 256); }));
  std::printf("LETHE kernel mult16 %u\n", (unsigned)best_of(20, [&] { dspm_mult_f32_ansi(A, Bm, Cm, 16, 16, 16); }));
  static float in[1024], out[1024];
  for (int i = 0; i < 1024; i++) in[i] = std::sin(0.05f * i);
  std::printf("LETHE kernel biquad1024 %u\n", (unsigned)best_of(20, [&] { w[0] = w[1] = 0; dsps_biquad_f32_ansi(in, out, 1024, coef, w); }));
  // libm calls of a step, per call (256 calls on varied arguments, minus the same loop without the call)
  for (int i = 0; i < 256; i++) a[i] = -4.0f + 8.0f * i / 256.0f, b[i] = 0.05f + 0.9f * i / 256.0f;
  auto per = [&](const char* name, auto f) {
    const uint32_t base = best_of(10, [&] { float s = 0; for (int i = 0; i < 256; i++) s += a[i]; sink = s; });
    const uint32_t t = best_of(10, [&] { float s = 0; for (int i = 0; i < 256; i++) s += f(i); sink = s; });
    std::printf("LETHE call %s %.1f\n", name, (t - base) / 256.0);
  };
  per("tanhf", [&](int i) { return std::tanh(a[i]); });
  per("expf", [&](int i) { return std::exp(a[i]); });
  per("powf", [&](int i) { return std::pow(b[i], 0.5f); });
  per("sqrtf", [&](int i) { return std::sqrt(b[i]); });
  per("fdiv", [&](int i) { return a[i] / b[i]; });
  per("fmul", [&](int i) { return a[i] * b[i]; });
  per("tanh", [&](int i) { return (float)std::tanh((double)a[i]); });
  per("dmul", [&](int i) { return (float)((double)a[i] * (double)b[i]); });
}

template <typename T>
static void replay(const lethe::Model& m, const char* name) {
  int32_t d[2];
  std::memcpy(d, trace_bin, sizeof d);
  const int n = d[0], N = d[1];
  size_t before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  lethe::Policy<T> pol(m);
  size_t after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  std::vector<uint32_t> cyc;
  cyc.reserve((size_t)n * N);
  double err = 0.0;
  for (int i = 0; i < n; i++) {
    pol.reset();
    for (int k = 0; k < N; k++) {
      const T y = T(trace_at(i, 0, k)), t = T(trace_at(i, 1, k)), tp = T(trace_at(i, 1, std::max(k - 1, 0)));
      const uint32_t c0 = cc();
      const T u = pol.step(y, t, tp);
      cyc.push_back(cc() - c0);
      err = std::max(err, std::abs(double(u) - trace_at(i, 2, k)));
    }
  }
  std::sort(cyc.begin(), cyc.end());
  std::printf("LETHE replay %s: %d patients x %d steps, max |u - u_jax| %.3e, counts per step min %u median %u p99 %u max %u, policy %u bytes\n",
              name, n, N, err, (unsigned)cyc[0], (unsigned)cyc[cyc.size() / 2], (unsigned)cyc[cyc.size() * 99 / 100], (unsigned)cyc.back(),
              (unsigned)(before - after));
}

static void rd(void* p, size_t n) {
  uint8_t* q = (uint8_t*)p;
  while (n) {
    const int r = uart_read_bytes(UART_NUM_0, q, n, portMAX_DELAY);
    if (r > 0) { q += r; n -= r; }
  }
}

template <typename T>
static void closed_loop(const lethe::Model& m) {
  std::unique_ptr<lethe::Policy<T>> pol;
  try {
    pol.reset(new lethe::Policy<T>(m));
  } catch (const std::bad_alloc&) {
    const float nan = NAN;
    uart_write_bytes(UART_NUM_0, "N", 1);
    (void)nan;
    return;
  }
  uart_write_bytes(UART_NUM_0, "K", 1);
  for (;;) {
    uint8_t c;
    rd(&c, 1);
    if (c != 'S') return;
    uint8_t b[16];
    rd(b, 16);
    int32_t k;
    float x[3];
    std::memcpy(&k, b, 4);
    std::memcpy(x, b + 4, 12);
    if (k == 0) pol->reset();
    const float u = (float)pol->step(T(x[0]), T(x[1]), T(x[2]));
    uart_write_bytes(UART_NUM_0, (const char*)&u, 4);
  }
}

extern "C" void app_main(void) {
  esp_log_level_set("*", ESP_LOG_NONE);
  uart_driver_install(UART_NUM_0, 2048, 0, 0, NULL, 0);
  uart_set_rx_full_threshold(UART_NUM_0, 1);   // wake the reader on every byte: the packets are tiny
  uart_set_rx_timeout(UART_NUM_0, 1);
  esp_chip_info_t ci;
  esp_chip_info(&ci);
  try {
    lethe::Model m = lethe::Model::parse(model_bin, model_end);
    std::printf("LETHE READY chip %d\n", (int)ci.model);
    std::fflush(stdout);
    for (;;) {
      uint8_t c;
      rd(&c, 1);
      if (c == 'B') {
        calibrate();
        replay<float>(m, "float");
        try { replay<double>(m, "double"); } catch (const std::bad_alloc&) { std::printf("LETHE replay double: does not fit\n"); }
        kernels();
        std::printf("LETHE DONE\n");
        std::fflush(stdout);
      } else if (c == 'F') {
        closed_loop<float>(m);
      } else if (c == 'D') {
        closed_loop<double>(m);
      }
    }
  } catch (const std::exception& e) {
    std::printf("LETHE error: %s\n", e.what());
  }
}
