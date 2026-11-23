// bench.cpp -- correctness and latency of the C++ inference engine.
//   bench <model.bin> <trace.bin> [repeats]
// trace.bin (export_cpp.py): int32 n_patients, int32 n_steps, then per patient float64 y[n], target[n], u_ref[n] (the JAX policy's outputs
// in a closed loop). The engine replays the recorded measurements / targets and is compared step by step with u_ref; then the 360-step
// episode is timed `repeats` times. Prints one JSON line.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>
#include "lethe_infer.hpp"

struct Trace { int n = 0, N = 0; std::vector<std::vector<double>> y, tgt, u; };

static Trace load_trace(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open trace");
  Trace t;
  int32_t d[2];
  f.read(reinterpret_cast<char*>(d), sizeof d);
  t.n = d[0]; t.N = d[1];
  for (int i = 0; i < t.n; i++)
    for (auto* v : {&t.y, &t.tgt, &t.u}) {
      v->emplace_back(t.N);
      f.read(reinterpret_cast<char*>(v->back().data()), t.N * sizeof(double));
    }
  if (!f) throw std::runtime_error("truncated trace");
  return t;
}

template <typename T>
static void run(const lethe::Model& m, const Trace& tr, int reps, const char* name, bool last) {
  lethe::Policy<T> pol(m);
  double err = 0.0;
  for (int i = 0; i < tr.n; i++) {
    pol.reset();
    for (int k = 0; k < tr.N; k++) {
      T u = pol.step(T(tr.y[i][k]), T(tr.tgt[i][k]), T(tr.tgt[i][std::max(k - 1, 0)]));
      err = std::max(err, std::abs(double(u) - tr.u[i][k]));
    }
  }
  std::vector<double> ns;
  volatile T sink = 0;
  for (int r = 0; r < reps; r++) {
    const int i = r % tr.n;
    pol.reset();
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < tr.N; k++) sink = sink + pol.step(T(tr.y[i][k]), T(tr.tgt[i][k]), T(tr.tgt[i][std::max(k - 1, 0)]));
    auto t1 = std::chrono::steady_clock::now();
    ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / tr.N);
  }
  std::sort(ns.begin(), ns.end());
  std::printf("\"%s\": {\"max_abs_err_vs_jax\": %.3e, \"ns_per_step_median\": %.1f, \"ns_per_step_p99\": %.1f}%s", name, err,
              ns[ns.size() / 2], ns[std::min(ns.size() - 1, ns.size() * 99 / 100)], last ? "" : ", ");
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: bench model.bin trace.bin [repeats]\n"); return 2; }
  const int reps = argc > 3 ? std::atoi(argv[3]) : 2000;
  lethe::Model m = lethe::Model::load(argv[1]);
  Trace tr = load_trace(argv[2]);
  std::printf("{\"H\": %d, \"NS\": %d, \"L\": %d, \"head\": %d, \"modes\": %d, ", m.H, m.NS, m.L, m.HD, m.P);
  run<float>(m, tr, reps, "float", false);
  run<double>(m, tr, reps, "double", true);
  std::printf("}\n");
  return 0;
}
