// trt_bench.cpp -- the TensorRT engine of one control step (Jetson Nano, TensorRT 8.2) as a controller: replays trace.bin (the JAX policy's
// closed loop, lethe.deploy.cpp), compares u step by step, times every step.
//   trt_bench <policy.plan> <trace.bin> [repeats] [graph]
// The recurrent states and u_prev stay on the GPU: two buffer sets, inputs and outputs swapped every step (ping-pong); a step copies y,
// tgt, tgt_prev in (one pinned 516-byte copy) and u out. graph = 1: each parity's copy + enqueue + copy captured once as a CUDA graph.
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)

struct Log : nvinfer1::ILogger {
  void log(Severity s, const char* m) noexcept override { if (s <= Severity::kWARNING) std::fprintf(stderr, "[trt] %s\n", m); }
} glog;

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

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: trt_bench policy.plan trace.bin [repeats] [graph]\n"); return 2; }
  const int reps = argc > 3 ? std::atoi(argv[3]) : 200;
  const bool graph = argc > 4 && std::atoi(argv[4]);
  Trace tr = load_trace(argv[2]);
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<char> plan((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto* rt = nvinfer1::createInferRuntime(glog);
  auto* eng = rt->deserializeCudaEngine(plan.data(), plan.size());
  if (!eng) { std::fprintf(stderr, "cannot deserialize %s\n", argv[1]); return 1; }
  auto* ctx = eng->createExecutionContext();
  const int nb = eng->getNbBindings();
  std::map<std::string, int> idx;
  std::vector<size_t> vol(nb);
  for (int b = 0; b < nb; b++) {
    idx[eng->getBindingName(b)] = b;
    auto d = eng->getBindingDimensions(b);
    size_t v = 1;
    for (int i = 0; i < d.nbDims; i++) v *= d.d[i];
    vol[b] = v;
    if (eng->getBindingDataType(b) != nvinfer1::DataType::kFLOAT) { std::fprintf(stderr, "binding %s not float\n", eng->getBindingName(b)); return 1; }
  }
  // the small inputs: one device block, 64 floats apart (256-byte aligned)
  float *din, *hin, *hu;
  CK(cudaMalloc(&din, 3 * 64 * sizeof(float)));
  CK(cudaMallocHost(&hin, 3 * 64 * sizeof(float)));
  CK(cudaMallocHost(&hu, sizeof(float)));
  std::memset(hin, 0, 3 * 64 * sizeof(float));
  const char* states[] = {"phys_re", "phys_im", "s4d_re", "s4d_im"};
  // ping-pong sets: set[p][s] (s = 0..3 the states, 4 = u); parity p reads set p, writes set 1 - p
  float* set[2][5];
  for (int p = 0; p < 2; p++) {
    for (int s = 0; s < 4; s++) CK(cudaMalloc(&set[p][s], vol[idx[states[s]]] * sizeof(float)));
    CK(cudaMalloc(&set[p][4], sizeof(float)));
  }
  std::vector<void*> bind[2];
  for (int p = 0; p < 2; p++) {
    bind[p].assign(nb, nullptr);
    bind[p][idx["y"]] = din; bind[p][idx["tgt"]] = din + 64; bind[p][idx["tgt_prev"]] = din + 128;
    bind[p][idx["u_prev"]] = set[p][4];
    bind[p][idx["u"]] = set[1 - p][4];
    for (int s = 0; s < 4; s++) {
      bind[p][idx[states[s]]] = set[p][s];
      bind[p][idx[std::string(states[s]) + "_out"]] = set[1 - p][s];
    }
  }
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  auto zero = [&]() { for (int s = 0; s < 5; s++) CK(cudaMemsetAsync(set[0][s], 0, (s < 4 ? vol[idx[states[s]]] : 1) * sizeof(float), st)); };
  auto enqueue = [&](int p) {
    CK(cudaMemcpyAsync(din, hin, (2 * 64 + 1) * sizeof(float), cudaMemcpyHostToDevice, st));
    if (!ctx->enqueueV2(bind[p].data(), st, nullptr)) { std::fprintf(stderr, "enqueue failed\n"); std::exit(1); }
    CK(cudaMemcpyAsync(hu, set[1 - p][4], sizeof(float), cudaMemcpyDeviceToHost, st));
  };
  cudaGraphExec_t ge[2];
  if (graph) {
    enqueue(0); enqueue(1); CK(cudaStreamSynchronize(st));            // TensorRT's lazy setup before capture
    for (int p = 0; p < 2; p++) {
      cudaGraph_t g;
      CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      enqueue(p);
      CK(cudaStreamEndCapture(st, &g));
      CK(cudaGraphInstantiate(&ge[p], g, nullptr, nullptr, 0));
    }
  }
  auto step = [&](int k, const std::vector<double>& y, const std::vector<double>& tgt) {
    hin[0] = float(y[k]); hin[64] = float(tgt[k]); hin[128] = float(tgt[std::max(k - 1, 0)]);
    const int p = k & 1;
    if (graph) CK(cudaGraphLaunch(ge[p], st)); else enqueue(p);
    CK(cudaStreamSynchronize(st));
    return double(*hu);
  };
  double err = 0;
  for (int i = 0; i < tr.n; i++) {
    zero();
    for (int k = 0; k < tr.N; k++) err = std::max(err, std::abs(step(k, tr.y[i], tr.tgt[i]) - tr.u[i][k]));
  }
  std::vector<double> ns;
  for (int r = 0; r < reps; r++) {
    const int i = r % tr.n;
    zero();
    CK(cudaStreamSynchronize(st));
    for (int k = 0; k < tr.N; k++) {
      auto t0 = std::chrono::steady_clock::now();
      step(k, tr.y[i], tr.tgt[i]);
      ns.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
    }
  }
  std::sort(ns.begin(), ns.end());
  double mean = 0; for (double v : ns) mean += v; mean /= ns.size();
  std::printf("{\"engine\": \"%s\", \"graph\": %d, \"max_abs_err_vs_jax\": %.3e, \"us_per_step_mean\": %.2f, \"us_per_step_median\": %.2f, "
              "\"us_per_step_p99\": %.2f, \"us_per_step_max\": %.2f, \"steps\": %zu}\n", argv[1], int(graph), err, mean / 1e3, ns[ns.size() / 2] / 1e3,
              ns[std::min(ns.size() - 1, ns.size() * 99 / 100)] / 1e3, ns.back() / 1e3, ns.size());
  return 0;
}
