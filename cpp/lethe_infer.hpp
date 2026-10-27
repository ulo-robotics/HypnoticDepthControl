// lethe_infer.hpp -- dependency-free C++17 inference engine of the Lethe policy (one patient, one control step at a time).
//
//   lethe::Model m = lethe::Model::load("model.bin");     // written by export_cpp.py
//   lethe::Policy<float> pol(m);                           // or Policy<double>
//   pol.reset();
//   for each 5 s step k:  u = pol.step(k, bis_measured, target_now, target_previous);    // infusion, mg/min, in [0, 100]
//
// The math is policy.act (JAX): the physics compensator (complex modes in target-normalized units), the encoder, L S4D layers (diagonal
// complex recurrence, GELU, gated mixing, residual, layer norm), the optional MLP head, the decoder. All matrices are discretized by the
// exporter (float64), so a step is only multiply-adds plus a few tanh / exp / sqrt.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lethe {

constexpr double DT = 5.0 / 60.0, T_END = 30.0, UMAX = 100.0, E0 = 100.0, EMAX = 100.0, GAMMA = 2.0, C50N = 4.5;

struct Layer {
  std::vector<double> dAr, dAi, dBr, dBi, Cr, Ci;    // (H, NS)
  std::vector<double> D, W, b, g, beta;               // D (H), W (H, 2H) row major, b (2H), g / beta (H)
};

struct Model {
  int H = 0, NS = 0, L = 0, HD = 0, P = 0;
  double u0 = 0, dec_b = 0;
  std::vector<double> lr, li, Bur, Bui, Bwr, Bwi, Cor, Coi;   // physics modes (P)
  std::vector<double> enc, enc_b;                             // enc (4, H) row major
  std::vector<Layer> layers;
  std::vector<double> hW, hb, dec;                            // hW (H, HD), hb (HD), dec (HD, or H without a head)

  static Model load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "LETHE01\0", 8) != 0) throw std::runtime_error(path + ": not a Lethe model file");
    int32_t d[5];
    f.read(reinterpret_cast<char*>(d), sizeof d);
    Model m;
    m.H = d[0]; m.NS = d[1]; m.L = d[2]; m.HD = d[3]; m.P = d[4];
    auto rd = [&](size_t n) {
      std::vector<double> v(n);
      f.read(reinterpret_cast<char*>(v.data()), n * sizeof(double));
      if (!f) throw std::runtime_error(path + ": truncated");
      return v;
    };
    const size_t H = m.H, NS = m.NS, P = m.P, HD = m.HD;
    auto s = rd(2); m.u0 = s[0]; m.dec_b = s[1];
    m.lr = rd(P); m.li = rd(P); m.Bur = rd(P); m.Bui = rd(P); m.Bwr = rd(P); m.Bwi = rd(P); m.Cor = rd(P); m.Coi = rd(P);
    m.enc = rd(4 * H); m.enc_b = rd(H);
    for (int l = 0; l < m.L; l++) {
      Layer y;
      y.dAr = rd(H * NS); y.dAi = rd(H * NS); y.dBr = rd(H * NS); y.dBi = rd(H * NS); y.Cr = rd(H * NS); y.Ci = rd(H * NS);
      y.D = rd(H); y.W = rd(H * 2 * H); y.b = rd(2 * H); y.g = rd(H); y.beta = rd(H);
      m.layers.push_back(std::move(y));
    }
    if (HD > 0) { m.hW = rd(H * HD); m.hb = rd(HD); }
    m.dec = rd(HD > 0 ? HD : H);
    return m;
  }
};

template <typename T>
class Policy {
 public:
  explicit Policy(const Model& m) : m_(m) {
    const size_t H = m.H, NS = m.NS;
    auto cv = [](const std::vector<double>& v) { return std::vector<T>(v.begin(), v.end()); };
    lr_ = cv(m.lr); li_ = cv(m.li); Bur_ = cv(m.Bur); Bui_ = cv(m.Bui); Bwr_ = cv(m.Bwr); Bwi_ = cv(m.Bwi); Cor_ = cv(m.Cor); Coi_ = cv(m.Coi);
    enc_ = cv(m.enc); enc_b_ = cv(m.enc_b); hW_ = cv(m.hW); hb_ = cv(m.hb); dec_ = cv(m.dec);
    for (auto& y : m.layers) {
      L_.push_back({cv(y.dAr), cv(y.dAi), cv(y.dBr), cv(y.dBi), cv(y.Cr), cv(y.Ci), cv(y.D), cv(y.W), cv(y.b), cv(y.g), cv(y.beta)});
    }
    hr_.assign(m.L, std::vector<T>(H * NS)); hi_.assign(m.L, std::vector<T>(H * NS));
    x_.resize(H); a_.resize(H); z_.resize(2 * H); hid_.resize(m.HD);
    reset();
  }

  void reset() {
    pr_.assign(m_.P, T(0)); pi_.assign(m_.P, T(0));
    for (auto& v : hr_) std::fill(v.begin(), v.end(), T(0));
    for (auto& v : hi_) std::fill(v.begin(), v.end(), T(0));
    u_prev_ = T(0);
  }

  // one control step: step index k (t = k * 5 s), measured BIS y, target now / at the previous step. Returns the infusion (mg/min).
  T step(int k, T y, T tgt, T tgt_prev) {
    const int H = m_.H, NS = m_.NS, P = m_.P, HD = m_.HD;
    const T r = wref(tgt) / T(C50N), rho = wref(tgt_prev) / wref(tgt);
    for (int p = 0; p < P; p++) { pr_[p] *= rho; pi_[p] *= rho; }
    T s = std::clamp((T(E0) - y) / T(EMAX), T(1e-3), T(1) - T(1e-3));
    const T w = std::min(T(C50N) * std::pow(s / (T(1) - s), T(1) / T(GAMMA)), T(3 * C50N));
    const T f[4] = {(y - tgt) / T(50), u_prev_ / (r * T(UMAX)), T(k * DT / T_END), w / (r * T(C50N)) - T(1)};
    for (int h = 0; h < H; h++) x_[h] = f[0] * enc_[h] + f[1] * enc_[H + h] + f[2] * enc_[2 * H + h] + f[3] * enc_[3 * H + h] + enc_b_[h];
    for (int l = 0; l < m_.L; l++) layer(l);
    T o = T(m_.dec_b);
    if (HD > 0) {
      for (int j = 0; j < HD; j++) hid_[j] = hb_[j];
      for (int i = 0; i < H; i++) {
        const T xi = x_[i];
        const T* row = &hW_[(size_t)i * HD];
        for (int j = 0; j < HD; j++) hid_[j] += xi * row[j];
      }
      for (int j = 0; j < HD; j++) o += gelu(hid_[j]) * dec_[j];
    } else {
      for (int h = 0; h < H; h++) o += x_[h] * dec_[h];
    }
    T ul = T(m_.u0);
    for (int p = 0; p < P; p++) ul += Cor_[p] * pr_[p] - Coi_[p] * pi_[p];
    const T u = std::clamp(r * (ul + T(10) * o), T(0), T(UMAX));
    const T un = u / r, wn = w / r;
    for (int p = 0; p < P; p++) {
      const T nr = lr_[p] * pr_[p] - li_[p] * pi_[p] + Bur_[p] * un + Bwr_[p] * wn;
      const T ni = lr_[p] * pi_[p] + li_[p] * pr_[p] + Bui_[p] * un + Bwi_[p] * wn;
      pr_[p] = nr; pi_[p] = ni;
    }
    u_prev_ = u;
    return u;
  }

 private:
  struct L_t { std::vector<T> dAr, dAi, dBr, dBi, Cr, Ci, D, W, b, g, beta; };

  static T wref(T tgt) {
    const T s = (T(E0) - tgt) / T(EMAX);
    return T(C50N) * std::pow(s / (T(1) - s), T(1) / T(GAMMA));
  }
  static T gelu(T x) {      // tanh approximation, as jax.nn.gelu
    const T c = T(0.7978845608028654);
    return T(0.5) * x * (T(1) + std::tanh(c * (x + T(0.044715) * x * x * x)));
  }

  void layer(int l) {
    const int H = m_.H, NS = m_.NS;
    const L_t& y = L_[l];
    T* hr = hr_[l].data();
    T* hi = hi_[l].data();
    for (int h = 0; h < H; h++) {                                  // S4D recurrence and output
      const T u = x_[h];
      T acc = T(0);
      for (int n = 0; n < NS; n++) {
        const int i = h * NS + n;
        const T nr = y.dAr[i] * hr[i] - y.dAi[i] * hi[i] + y.dBr[i] * u;
        const T ni = y.dAr[i] * hi[i] + y.dAi[i] * hr[i] + y.dBi[i] * u;
        hr[i] = nr; hi[i] = ni;
        acc += nr * y.Cr[i] - ni * y.Ci[i];
      }
      a_[h] = gelu(T(2) * acc + y.D[h] * u);
    }
    for (int j = 0; j < 2 * H; j++) z_[j] = y.b[j];                // gated mixing: z = a W + b
    for (int i = 0; i < H; i++) {
      const T ai = a_[i];
      const T* row = &y.W[(size_t)i * 2 * H];
      for (int j = 0; j < 2 * H; j++) z_[j] += ai * row[j];
    }
    T mean = T(0);
    for (int h = 0; h < H; h++) {
      x_[h] += z_[h] / (T(1) + std::exp(-z_[H + h]));
      mean += x_[h];
    }
    mean /= T(H);
    T var = T(0);
    for (int h = 0; h < H; h++) { x_[h] -= mean; var += x_[h] * x_[h]; }
    const T inv = T(1) / std::sqrt(var / T(H) + T(1e-5));
    for (int h = 0; h < H; h++) x_[h] = x_[h] * inv * y.g[h] + y.beta[h];
  }

  const Model& m_;
  std::vector<T> lr_, li_, Bur_, Bui_, Bwr_, Bwi_, Cor_, Coi_, enc_, enc_b_, hW_, hb_, dec_;
  std::vector<L_t> L_;
  std::vector<std::vector<T>> hr_, hi_;
  std::vector<T> pr_, pi_, x_, a_, z_, hid_;
  T u_prev_ = T(0);
};

}  // namespace lethe
