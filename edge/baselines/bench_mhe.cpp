// bench_mhe.cpp -- MHE+NMPC (lethe.baselines.mpc.MHENMPC, its defaults) on the edge boards: per step, one MHE per delay hypothesis and one
// robust NMPC, each IPOPT over the CasADi-generated NLP functions (lethe.deploy.edge.baselines.mhe_nmpc), the glue ported from
// MHENMPC.act.
//   bench_mhe <trace.bin> [repeats]
// Replays the Python controller's closed loop (lethe.deploy.edge.baselines.reference_traces) and compares u step by step, then times
// `repeats` episodes. Prints one JSON line.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <IpStdCInterface.h>
extern "C" {
#include "mhe_nlp.h"
#include "nmpc_nlp.h"
}
#include "mhe_consts.h"
static const double UMAX = 100.0;

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

typedef int (*EvalFn)(const casadi_real**, casadi_real**, casadi_int*, casadi_real*, int);
typedef int (*WorkFn)(casadi_int*, casadi_int*, casadi_int*, casadi_int*);
struct CF {
  EvalFn fn; std::vector<const casadi_real*> arg; std::vector<casadi_real*> res; std::vector<casadi_int> iw; std::vector<casadi_real> w;
  CF(EvalFn f, WorkFn work) : fn(f) { casadi_int a, r, i, ww; work(&a, &r, &i, &ww); arg.resize(a); res.resize(r); iw.resize(i); w.resize(ww); }
  void operator()(const double* x, const double* p, double* out) {
    arg[0] = x; arg[1] = p; res[0] = out;
    fn(arg.data(), res.data(), iw.data(), w.data(), 0);
  }
};

// one IPOPT problem over generated f / grad / Hessian (lower triangle: sparse triplets, or dense when rows == nullptr)
struct NLP {
  CF F, G, H; int n, nnz; const int *hr, *hc; std::vector<double> p, hbuf; int it = 0; IpoptProblem prob = nullptr;
  NLP(EvalFn f, WorkFn fw, EvalFn g, WorkFn gw, EvalFn h, WorkFn hw, int n_, int np, int nnz_, const int* r, const int* c)
      : F(f, fw), G(g, gw), H(h, hw), n(n_), nnz(nnz_), hr(r), hc(c), p(np), hbuf(r ? nnz_ : n_ * n_) {}
  static Bool ef(Index, Number* x, Bool, Number* f, UserDataPtr u) { auto* s = (NLP*)u; s->F(x, s->p.data(), f); return TRUE; }
  static Bool eg(Index, Number* x, Bool, Number* g, UserDataPtr u) { auto* s = (NLP*)u; s->G(x, s->p.data(), g); return TRUE; }
  static Bool ec(Index, Number*, Bool, Index, Number*, UserDataPtr) { return TRUE; }
  static Bool ej(Index, Number*, Bool, Index, Index, Index*, Index*, Number*, UserDataPtr) { return TRUE; }
  static Bool eh(Index n, Number* x, Bool, Number of, Index, Number*, Bool, Index, Index* iR, Index* jC, Number* v, UserDataPtr u) {
    auto* s = (NLP*)u;
    if (!v) {
      if (s->hr) for (int e = 0; e < s->nnz; e++) { iR[e] = s->hr[e]; jC[e] = s->hc[e]; }
      else { int e = 0; for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) { iR[e] = i; jC[e] = j; e++; } }
      return TRUE;
    }
    s->H(x, s->p.data(), s->hbuf.data());
    if (s->hr) for (int e = 0; e < s->nnz; e++) v[e] = of * s->hbuf[e];
    else { int e = 0; for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) v[e++] = of * s->hbuf[i + n * j]; }
    return TRUE;
  }
  static Bool cb(Index, Index it, Number, Number, Number, Number, Number, Number, Number, Number, Index, UserDataPtr u) {
    ((NLP*)u)->it = it; return TRUE;
  }
  // IPOPT's C interface fixes the bounds when the problem is created: it is created once and re-created only when they change (the
  // MHE's during the first d + W steps, while its window reaches before the start)
  std::vector<double> lbc, ubc; bool adapt = false;
  ApplicationReturnStatus solve(double* x, const double* lb, const double* ub, bool adaptive, double* obj) {
    if (!prob || adaptive != adapt || !std::equal(lb, lb + n, lbc.begin()) || !std::equal(ub, ub + n, ubc.begin())) {
      if (prob) FreeIpoptProblem(prob);
      lbc.assign(lb, lb + n); ubc.assign(ub, ub + n); adapt = adaptive;
      prob = CreateIpoptProblem(n, lbc.data(), ubc.data(), 0, nullptr, nullptr, 0, hr ? nnz : n * (n + 1) / 2, 0, ef, ec, eg, ej, eh);
      AddIpoptIntOption(prob, (char*)"print_level", 0);
      AddIpoptStrOption(prob, (char*)"sb", (char*)"yes");
      AddIpoptNumOption(prob, (char*)"tol", TOL);
      AddIpoptIntOption(prob, (char*)"max_iter", MAXIT);
      if (adaptive) AddIpoptStrOption(prob, (char*)"mu_strategy", (char*)"adaptive");
      SetIntermediateCallback(prob, cb);
    }
    it = 0;
    return IpoptSolve(prob, x, nullptr, obj, nullptr, nullptr, nullptr, this);
  }
};

static void f_np(const double* x, double u, double a, double* o) {   // one model step: nominal compartments, effect site with factor a
  double c[3];
  for (int i = 0; i < 3; i++) c[i] = A3[3 * i] * x[0] + A3[3 * i + 1] * x[1] + A3[3 * i + 2] * x[2] + B3[i] * u;
  o[3] = a * x[3] + (1 - a) * 0.5 * (x[0] + c[0]);
  o[0] = c[0]; o[1] = c[1]; o[2] = c[2];
}

struct MHE {
  static const int NX5 = 5 * (MW + 1);
  std::vector<NLP> mhes;                                    // one problem per delay hypothesis (their bounds differ while warming up)
  NLP mpc{nmpc_f, nmpc_f_work, nmpc_g, nmpc_g_work, nmpc_h, nmpc_h_work, MNBLK, MNP2, 0, nullptr, nullptr};
  double th[MNOBS][3], X[MNOBS][MW + 1][5], w[MNBLK], uprev;
  std::vector<double> yh, uh;
  long iters = 0, solves = 0, fails = 0;
  MHE() { mhes.reserve(MNOBS); for (int m = 0; m < MNOBS; m++) mhes.emplace_back(mhe_f, mhe_f_work, mhe_g, mhe_g_work, mhe_h, mhe_h_work, MNX, MNP, MNNZ, MHR, MHC); }
  void reset(int N) {
    for (int m = 0; m < MNOBS; m++) {
      th[m][0] = LG0; th[m][1] = 0.0; th[m][2] = E0MEAN;
      for (int i = 0; i <= MW; i++) { for (int c = 0; c < 4; c++) X[m][i][c] = 0.0; X[m][i][4] = LC0; }
    }
    yh.assign(N, 0.0); uh.assign(N, 0.0);
    std::fill(w, w + MNBLK, 0.0); uprev = 0.0;
  }
  double act(int k, double y, double tgt, bool timed) {
    const int N = yh.size();
    yh[k] = y;
    double Jw[MNOBS];
    double sa[3];
    for (int i = 0; i < 3; i++) sa[i] = SARR1[i] + (SARR0[i] - SARR1[i]) * std::exp(-k / ARR_TAU);
    for (int m = 0; m < MNOBS; m++) {
      const int d = MDSET[m], t0 = k - d - MW;
      double a = std::exp(-std::exp(th[m][1]) * KE0 * DT);
      NLP& mhe = mhes[m];
      double* P = mhe.p.data();
      double xb[5];
      if (k > 0) std::memcpy(xb, X[m][1], sizeof xb);
      else { std::fill(xb, xb + 4, 0.0); xb[4] = LC0; }
      for (int c = 0; c < 4; c++) xb[c] = t0 >= 0 ? std::max(xb[c], 0.0) : 0.0;
      std::memcpy(P, xb, sizeof xb);
      double* uu = P + 5, * yy = P + 5 + MW, * mk = P + 5 + 2 * MW;
      for (int i = 0; i < MW; i++) {
        const int ut = t0 + i, rt = k - MW + 1 + i;
        uu[i] = ut >= 0 ? uh[ut] : 0.0;
        yy[i] = rt >= 0 ? yh[rt] : 0.0;
        mk[i] = rt >= 0 ? 1.0 : 0.0;
      }
      std::memcpy(P + 5 + 3 * MW, th[m], 3 * sizeof(double));
      std::memcpy(P + 8 + 3 * MW, sa, 3 * sizeof(double));
      P[11 + 3 * MW] = SY;
      double x0[MNX], lb[MNX], ub[MNX];
      for (int i = 0; i <= MW; i++) {                       // warm start: the previous window shifted, its last state predicted
        double* xi = x0 + 5 * i;
        if (k > 0) {
          if (i < MW) std::memcpy(xi, X[m][i + 1], 5 * sizeof(double));
          else { f_np(X[m][MW], uu[MW - 1], a, xi); xi[4] = X[m][MW][4]; }
        } else std::memcpy(xi, X[m][i], 5 * sizeof(double));
        const bool on = t0 + i >= 0;
        for (int c = 0; c < 4; c++) { lb[5 * i + c] = 0.0; ub[5 * i + c] = on ? 1e3 : 0.0; }
        lb[5 * i + 4] = LCLB; ub[5 * i + 4] = LCUB;
        for (int c = 0; c < 5; c++) xi[c] = std::min(std::max(xi[c], lb[5 * i + c]), ub[5 * i + c]);
      }
      for (int c = 0; c < 3; c++) { x0[NX5 + c] = th[m][c]; lb[NX5 + c] = THLB[c]; ub[NX5 + c] = THUB[c]; }
      double obj = 0;
      ApplicationReturnStatus rs = mhe.solve(x0, lb, ub, false, &obj);
      if (timed) { solves++; iters += mhe.it; fails += rs < 0; }
      bool ok = true;
      for (double v : x0) ok = ok && std::isfinite(v);
      if (ok) {
        for (int i = 0; i <= MW; i++) std::memcpy(X[m][i], x0 + 5 * i, 5 * sizeof(double));
        std::memcpy(th[m], x0 + NX5, 3 * sizeof(double));
      }
      Jw[m] = obj;                                          // MHENMPC uses r["f"] whatever the status
    }
    double jmin = *std::min_element(Jw, Jw + MNOBS), p[MNOBS], s = 0;
    for (int m = 0; m < MNOBS; m++) s += (p[m] = std::exp(-KAPPA * (Jw[m] - jmin)));
    double s2 = 0;
    for (int m = 0; m < MNOBS; m++) s2 += (p[m] = std::max(p[m] / s, PFLOOR));
    double* Pc = mpc.p.data();
    for (int m = 0; m < MNOBS; m++) {
      const int d = MDSET[m];
      double a = std::exp(-std::exp(th[m][1]) * KE0 * DT), xn[4], t[4];
      std::memcpy(xn, X[m][MW], sizeof xn);
      for (int st = 0; st < MDMAX; st++) {                  // the window's last estimate (model time k - d) over u_(k-d) .. u_(k-1)
        const int tt = k - d + st;
        if (tt < k && tt >= 0) { f_np(xn, uh[std::min(std::max(tt, 0), N - 1)], a, t); std::memcpy(xn, t, sizeof t); }
      }
      double* q = Pc + 9 * m;
      std::memcpy(q, xn, sizeof xn);
      q[4] = X[m][MW][4]; q[5] = th[m][0]; q[6] = a; q[7] = p[m] / s2; q[8] = th[m][2];
    }
    Pc[9 * MNOBS] = tgt; Pc[9 * MNOBS + 1] = uprev;
    double x[MNBLK], lb[MNBLK], ub[MNBLK], obj;
    std::memcpy(x, w, sizeof w);
    std::fill(lb, lb + MNBLK, 0.0); std::fill(ub, ub + MNBLK, UMAX);
    ApplicationReturnStatus rs = mpc.solve(x, lb, ub, true, &obj);
    if (timed) { solves++; iters += mpc.it; fails += rs < 0; }
    for (int j = 0; j < MNBLK; j++) if (std::isfinite(x[j])) w[j] = x[j];
    double u = std::min(std::max(w[0], 0.0), UMAX);
    uprev = u;
    uh[k] = u;
    return u;
  }
};

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: bench_mhe trace.bin [repeats]\n"); return 2; }
  Trace tr = load_trace(argv[1]);
  const int reps = argc > 2 ? std::atoi(argv[2]) : 8;
  MHE c;
  double err = 0;
  for (int i = 0; i < tr.n; i++) {
    c.reset(tr.N);
    for (int k = 0; k < tr.N; k++) err = std::max(err, std::abs(c.act(k, tr.y[i][k], tr.tgt[i][k], false) - tr.u[i][k]));
  }
  std::vector<double> ns;
  for (int r = 0; r < reps; r++) {
    const int i = r % tr.n;
    c.reset(tr.N);
    for (int k = 0; k < tr.N; k++) {
      auto t0 = std::chrono::steady_clock::now();
      volatile double u = c.act(k, tr.y[i][k], tr.tgt[i][k], true);
      (void)u;
      ns.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
    }
  }
  std::vector<double> sorted = ns;
  std::sort(sorted.begin(), sorted.end());
  double mean = 0; for (double v : ns) mean += v; mean /= ns.size();
  std::printf("{\"baseline\": \"mhe\", \"max_abs_err_vs_python\": %.3e, \"us_per_step_mean\": %.2f, \"us_per_step_median\": %.2f, "
              "\"us_per_step_p99\": %.2f, \"us_per_step_max\": %.2f, \"steps\": %zu, \"ipopt_iters_mean\": %.2f, \"ipopt_solves_per_step\": %d, "
              "\"ipopt_failures\": %ld}\n", err, mean / 1e3, sorted[sorted.size() / 2] / 1e3, sorted[std::min(sorted.size() - 1, sorted.size() * 99 / 100)] / 1e3,
              sorted.back() / 1e3, ns.size(), double(c.iters) / c.solves, MNOBS + 1, c.fails);
  return 0;
}
