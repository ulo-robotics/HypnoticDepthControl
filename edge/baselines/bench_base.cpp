// bench_base.cpp -- the CasADi-generated classical baselines on the edge boards: UKF-MM (one generated step) and MPC-MM (the generated
// UKF-MM estimator + IPOPT over the generated NLP functions, warm-started from the previous plan, as lethe.baselines.mpc.MPCMM).
//   bench_base <ukfmm|mpcmm> <trace.bin> [repeats]
// Replays the recorded measurements / targets of the Python controller's closed loop (lethe.deploy.edge.baselines.reference_traces)
// and compares u step by step; then times `repeats` episodes. Prints one JSON line: per-step latency (median, p99, max over all timed
// steps) and, for MPC-MM, IPOPT iterations.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <IpStdCInterface.h>
extern "C" {
#include "baselines.h"
}
#include "consts.h"

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

// one generated CasADi function with its work arrays
struct CF {
  int (*fn)(const casadi_real**, casadi_real**, casadi_int*, casadi_real*, int);
  std::vector<const casadi_real*> arg; std::vector<casadi_real*> res; std::vector<casadi_int> iw; std::vector<casadi_real> w;
  CF(int (*f)(const casadi_real**, casadi_real**, casadi_int*, casadi_real*, int),
     int (*work)(casadi_int*, casadi_int*, casadi_int*, casadi_int*)) : fn(f) {
    casadi_int a, r, i, ww;
    work(&a, &r, &i, &ww);
    arg.resize(a); res.resize(r); iw.resize(i); w.resize(ww);
  }
  void operator()(std::initializer_list<const double*> in, std::initializer_list<double*> out) {
    std::copy(in.begin(), in.end(), arg.begin());
    std::copy(out.begin(), out.end(), res.begin());
    fn(arg.data(), res.data(), iw.data(), w.data(), 0);
  }
};

struct UKFMM {
  CF f{ukfmm, ukfmm_work};
  double z[5 * NOBS], P[25 * NOBS], Rm[NOBS], ll[NOBS], uwin[NWIN];
  double xf[4 * NOBS], lc[NOBS], p[NOBS];
  void reset() {
    std::memcpy(z, Z0, sizeof z);
    for (int m = 0; m < NOBS; m++) { std::memcpy(P + 25 * m, P0, 25 * sizeof(double)); Rm[m] = R0; ll[m] = 0; }
    std::fill(uwin, uwin + NWIN, 0.0);
  }
  double estimate(double y, int k, double tgt) {           // updates the observers; returns UKF-MM's own (LQR mixture) input
    double kk = k, u;
    f({z, P, Rm, ll, &y, &kk, &tgt, uwin}, {z, P, Rm, ll, &u, xf, lc, p});   // in place: CasADi copies the inputs into its work first
    return u;
  }
  void push(double u) { std::memmove(uwin, uwin + 1, (NWIN - 1) * sizeof(double)); uwin[NWIN - 1] = u; }
};

// ---- MPC-MM: IPOPT over the generated cost / gradient / Hessian
struct MPC {
  CF F{mpcmm_f, mpcmm_f_work}, G{mpcmm_g, mpcmm_g_work}, H{mpcmm_h, mpcmm_h_work};
  double par[6 * NOBS + 2], Hd[NBLK * NBLK];
  int it = 0;                                              // IPOPT iterations of the last solve
};
static Bool ev_f(Index, Number* x, Bool, Number* f, UserDataPtr ud) { auto* s = (MPC*)ud; s->F({x, s->par}, {f}); return TRUE; }
static Bool ev_g(Index, Number* x, Bool, Number* g, UserDataPtr ud) { auto* s = (MPC*)ud; s->G({x, s->par}, {g}); return TRUE; }
static Bool ev_c(Index, Number*, Bool, Index, Number*, UserDataPtr) { return TRUE; }
static Bool ev_j(Index, Number*, Bool, Index, Index, Index*, Index*, Number*, UserDataPtr) { return TRUE; }
static Bool ev_h(Index n, Number* x, Bool, Number of, Index, Number*, Bool, Index, Index* iR, Index* jC, Number* v, UserDataPtr ud) {
  int e = 0;
  if (!v) { for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) { iR[e] = i; jC[e] = j; e++; } return TRUE; }
  auto* s = (MPC*)ud;
  s->H({x, s->par}, {s->Hd});
  for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) v[e++] = of * s->Hd[i + n * j];
  return TRUE;
}

struct Stats { std::vector<double> ns; double err = 0; long iters = 0, solves = 0, fails = 0; };

template <class Step>
static void replay(const Trace& tr, int reps, Stats& st, Step&& step) {
  for (int i = 0; i < tr.n; i++) {                         // correctness: every patient once
    step.reset();
    for (int k = 0; k < tr.N; k++) st.err = std::max(st.err, std::abs(step(tr.y[i][k], k, tr.tgt[i][k], false) - tr.u[i][k]));
  }
  for (int r = 0; r < reps; r++) {                         // latency: per step
    const int i = r % tr.n;
    step.reset();
    for (int k = 0; k < tr.N; k++) {
      auto t0 = std::chrono::steady_clock::now();
      volatile double u = step(tr.y[i][k], k, tr.tgt[i][k], true);
      (void)u;
      st.ns.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: bench_base ukfmm|mpcmm trace.bin [repeats]\n"); return 2; }
  const std::string which = argv[1];
  Trace tr = load_trace(argv[2]);
  const int reps = argc > 3 ? std::atoi(argv[3]) : (which == "ukfmm" ? 2000 : 40);
  Stats st;
  if (which == "ukfmm") {
    struct S { UKFMM e; void reset() { e.reset(); }
      double operator()(double y, int k, double tgt, bool) { double u = e.estimate(y, k, tgt); e.push(u); return u; } } s;
    replay(tr, reps, st, s);
  } else if (which == "mpcmm") {
    struct S {
      UKFMM e; MPC m; IpoptProblem nlp = nullptr; double w[NBLK], lb[NBLK], ub[NBLK], uprev = 0; Stats* st;
      void reset() {
        e.reset(); uprev = 0; std::fill(w, w + NBLK, 0.0);
        if (!nlp) {
          std::fill(lb, lb + NBLK, 0.0); std::fill(ub, ub + NBLK, UMAX);
          nlp = CreateIpoptProblem(NBLK, lb, ub, 0, nullptr, nullptr, 0, NBLK * (NBLK + 1) / 2, 0, ev_f, ev_c, ev_g, ev_j, ev_h);
          AddIpoptIntOption(nlp, (char*)"print_level", 0);
          AddIpoptStrOption(nlp, (char*)"sb", (char*)"yes");
          AddIpoptNumOption(nlp, (char*)"tol", 1e-6);
          AddIpoptIntOption(nlp, (char*)"max_iter", 100);
          AddIpoptStrOption(nlp, (char*)"mu_strategy", (char*)"adaptive");
          SetIntermediateCallback(nlp, [](Index, Index it, Number, Number, Number, Number, Number, Number, Number, Number, Index,
                                          UserDataPtr ud) -> Bool { ((MPC*)ud)->it = it; return TRUE; });
        }
      }
      double operator()(double y, int k, double tgt, bool timed) {
        e.estimate(y, k, tgt);
        double* p = m.par;
        std::memcpy(p, e.xf, sizeof e.xf); p += 4 * NOBS;
        std::memcpy(p, e.lc, sizeof e.lc); p += NOBS;
        std::memcpy(p, e.p, sizeof e.p); p += NOBS;
        p[0] = tgt; p[1] = uprev;
        double x[NBLK], obj, zl[NBLK], zu[NBLK];
        std::memcpy(x, w, sizeof w);
        UserDataPtr ud = &m;
        m.it = 0;
        ApplicationReturnStatus rs = IpoptSolve(nlp, x, nullptr, &obj, nullptr, zl, zu, ud);
        bool ok = true;
        for (double v : x) ok = ok && std::isfinite(v);
        if (ok) std::memcpy(w, x, sizeof w);
        else { std::memmove(w, w + 1, (NBLK - 1) * sizeof(double)); }  // the previous plan shifted (MPCMM's fallback)
        if (timed) { st->solves++; st->iters += m.it; st->fails += (rs < 0); }
        double u = std::min(std::max(w[0], 0.0), UMAX);
        uprev = u;
        e.push(u);
        return u;
      }
    } s;
    s.st = &st;
    replay(tr, reps, st, s);
  } else { std::fprintf(stderr, "unknown baseline %s\n", which.c_str()); return 2; }
  std::vector<double> ns = st.ns;
  std::sort(ns.begin(), ns.end());
  double mean = 0; for (double v : ns) mean += v; mean /= ns.size();
  std::printf("{\"baseline\": \"%s\", \"max_abs_err_vs_python\": %.3e, \"us_per_step_mean\": %.2f, \"us_per_step_median\": %.2f, "
              "\"us_per_step_p99\": %.2f, \"us_per_step_max\": %.2f, \"steps\": %zu", which.c_str(), st.err, mean / 1e3,
              ns[ns.size() / 2] / 1e3, ns[std::min(ns.size() - 1, ns.size() * 99 / 100)] / 1e3, ns.back() / 1e3, ns.size());
  if (st.solves) std::printf(", \"ipopt_iters_mean\": %.2f, \"ipopt_failures\": %ld", double(st.iters) / st.solves, st.fails);
  std::printf("}\n");
  return 0;
}
