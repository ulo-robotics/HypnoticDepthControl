// cbwd.cu -- the adjoint (backward) sweep of kstep.py as one CUDA kernel, ported from the old meta_opt engine (cengine.py + cuda/cbwd.cu). The
// math is kstep.bwd_core (the VJP of kstep.step, masked or plain layout) with the blocked S4D adjoint of kstep._backward_blocked, written out by
// hand. Compile-time configuration: the generated header cfg.h (cbwd.py).
//
// Layout: one thread block = PB patients of one network (blockIdx.y = member); LANES threads per patient, thread `lane` owns channels
// lane + LANES j. Per-patient scalars are computed redundantly by the patient's lanes (butterfly all-reduces: every lane gets the same bits).
// Network and patient constants are staged in shared memory once. The GLU / head matmuls and their transposes are block-wide with the block's
// patients as rows, on tensor cores (mma.sync m16n8k8 TF32 in the 3xTF32 split form, close to FP32), operands staged in shared memory.
// Every input and output has a leading member axis (cbwd.py calls the kernel with vmap_method="broadcast_all").
// Outputs: with LAMOUT = 1 lam (carry adjoint after every step) and e (kstep.xla_backward's contract); always t_x (B, N, 2 LP HP): per step the
// S4D layer inputs u (GRADS) and output cotangents cy, layer l at columns l 2HP and l 2HP + HP (the layout cuda/s4grad.cu reads), and two scratch
// buffers of the blocked S4D adjoint; with GRADS = 1 also the per-patient gradients that are sums over steps (accumulated in registers) and the
// tapes the large matrices' gradients need (a, dzv, dzg per layer, the trunk output xe, the head pre-activation cotangent dpre).
#include <cuda_runtime.h>
#include <stdint.h>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

#define NT (PB * LANES)
#define CH (HP / LANES)
#define CHD ((HDP + LANES - 1) / LANES)
#define CP5 ((P5P + LANES - 1) / LANES)
#define FULL 0xffffffffu
#define LDS ((2 * HP) > HDP ? (2 * HP) : HDP)
#define S_TAPE (6 + 2 * P5P + DPAD)
#define DPL ((DPAD + LANES - 1) / LANES)                      // delay-line entries per lane: entry lane + LANES j

#define NSM (6 * HP + HDP + 8 * P5P + 2 + 2 * LP * HP)          // packed per-patient small gradients (GRADS)
#define WX (2 * LP * HP)                                       // row of t_x

struct Args { const float* in[NIN]; float* out[NOUT]; };
#define INP(n) (A.in[I_##n])
#define LD(n, i) __ldg(A.in[I_##n] + (i))
#define OUT(n) (A.out[O_##n])

__device__ __forceinline__ float AR(float v) {
#pragma unroll
  for (int o = LANES / 2; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o, LANES);
  return v;
}
__device__ __forceinline__ float BC(float v, int src) { return __shfl_sync(FULL, v, src, LANES); }
__device__ __forceinline__ float d_max(float x, float r) { return x > r ? 1.f : (x == r ? 0.5f : 0.f); }   // JAX tie conventions
__device__ __forceinline__ float d_min(float x, float r) { return x < r ? 1.f : (x == r ? 0.5f : 0.f); }
__device__ __forceinline__ float d_clip(float x, float lo, float hi) { return d_max(x, lo) * d_min(fmaxf(lo, x), hi); }
__device__ __forceinline__ float relu(float x) { return x > 0.f ? x : 0.f; }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ void gelu_d(float x, float& y, float& d) {
  const float c = 0.7978845608028654f;
  float u = c * (x + 0.044715f * x * x * x);
  float th = tanhf(u);
  float cdf = 0.5f * (1.f + th);
  y = x * cdf;
  d = cdf + x * 0.5f * (1.f - th * th) * c * (1.f + 3.f * 0.044715f * x * x);
}

// Y[p][n] = sum_k X[p][k] W[k][n] (+ sum_k X2[p][k] W2[k][n]) + bias[n]; X = hi + lo staged in shared memory (row length LDS), W global.
template <int KD, int KD2, int ND>
__device__ __forceinline__ void mm(const float* X, const float* XL, const float* __restrict__ W, const float* X2, const float* XL2, const float* __restrict__ W2,
                                   const float* __restrict__ bias, float* Y) {
  const int t = threadIdx.x;
  if constexpr (ND >= LANES) {
    constexpr int OPT = ND / LANES;
    constexpr int TP = (OPT >= 2 && PB % 2 == 0) ? 2 : 1;
    constexpr int TN = OPT / TP;
    constexpr int NCT = ND / TN;
    const int nt = t % NCT, pt = t / NCT;
    float acc[TP][TN];
#pragma unroll
    for (int a = 0; a < TP; a++)
#pragma unroll
      for (int c = 0; c < TN; c++) acc[a][c] = bias ? __ldg(bias + nt + c * NCT) : 0.f;
#pragma unroll 8
    for (int k = 0; k < KD; k++) {
      float w[TN];
#pragma unroll
      for (int c = 0; c < TN; c++) w[c] = __ldg(W + k * ND + nt + c * NCT);
#pragma unroll
      for (int a = 0; a < TP; a++) {
        const float x = X[(pt * TP + a) * LDS + k] + XL[(pt * TP + a) * LDS + k];
#pragma unroll
        for (int c = 0; c < TN; c++) acc[a][c] = fmaf(x, w[c], acc[a][c]);
      }
    }
    if constexpr (KD2 > 0) {
#pragma unroll 8
      for (int k = 0; k < KD2; k++) {
        float w[TN];
#pragma unroll
        for (int c = 0; c < TN; c++) w[c] = __ldg(W2 + k * ND + nt + c * NCT);
#pragma unroll
        for (int a = 0; a < TP; a++) {
          const float x = X2[(pt * TP + a) * LDS + k] + XL2[(pt * TP + a) * LDS + k];
#pragma unroll
          for (int c = 0; c < TN; c++) acc[a][c] = fmaf(x, w[c], acc[a][c]);
        }
      }
    }
#pragma unroll
    for (int a = 0; a < TP; a++)
#pragma unroll
      for (int c = 0; c < TN; c++) Y[(pt * TP + a) * LDS + nt + c * NCT] = acc[a][c];
  } else {
    for (int o = t; o < PB * ND; o += NT) {
      const int p = o / ND, n = o % ND;
      float s = bias ? __ldg(bias + n) : 0.f;
      for (int k = 0; k < KD; k++) s = fmaf(X[p * LDS + k] + XL[p * LDS + k], __ldg(W + k * ND + n), s);
      for (int k = 0; k < KD2; k++) s = fmaf(X2[p * LDS + k] + XL2[p * LDS + k], __ldg(W2 + k * ND + n), s);
      Y[p * LDS + n] = s;
    }
  }
}

// tensor cores: x * w ~ hi_x hi_w + hi_x lo_w + lo_x hi_w (3xTF32); rows = the block's patients (padded to 16), one warp per 16 x 8 tile
__device__ __forceinline__ uint32_t tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ void mma8(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int KD>
__device__ __forceinline__ void tc_k(float d[4], const float* X, const float* XL, const float* __restrict__ W, int ND, int r0, int r1, bool v0, bool v1,
                                     int c0, int gid, int tq) {
#pragma unroll 4
  for (int k0 = 0; k0 < KD; k0 += 8) {
    const int i0 = r0 * LDS + k0 + tq, i1 = r1 * LDS + k0 + tq;
    uint32_t ah[4] = {v0 ? __float_as_uint(X[i0]) : 0u, v1 ? __float_as_uint(X[i1]) : 0u, v0 ? __float_as_uint(X[i0 + 4]) : 0u, v1 ? __float_as_uint(X[i1 + 4]) : 0u};
    uint32_t al[4] = {v0 ? __float_as_uint(XL[i0]) : 0u, v1 ? __float_as_uint(XL[i1]) : 0u, v0 ? __float_as_uint(XL[i0 + 4]) : 0u, v1 ? __float_as_uint(XL[i1 + 4]) : 0u};
    const int w0 = (k0 + tq) * ND + c0 + gid, w1 = w0 + 4 * ND;
    const float bf0 = __ldg(W + w0), bf1 = __ldg(W + w1);
    uint32_t bh[2] = {tf32(bf0), tf32(bf1)};
    uint32_t bl[2] = {tf32(bf0 - __uint_as_float(bh[0])), tf32(bf1 - __uint_as_float(bh[1]))};
    mma8(d, al, bh);
    mma8(d, ah, bl);
    mma8(d, ah, bh);
  }
}
template <int KD, int KD2, int ND>
__device__ __forceinline__ void mmx(const float* X, const float* XL, const float* W, const float* X2, const float* XL2, const float* W2,
                                    const float* __restrict__ bias, float* Y) {
  if constexpr (TC && ND % 8 == 0 && KD % 8 == 0 && KD2 % 8 == 0) {
    const int warp = threadIdx.x >> 5, ln = threadIdx.x & 31, gid = ln >> 2, tq = ln & 3;
    constexpr int NTL = ND / 8;
    constexpr int MT = (PB + 15) / 16;
    for (int tile = warp; tile < MT * NTL; tile += NT / 32) {
      const int r0 = (tile / NTL) * 16 + gid, c0 = (tile % NTL) * 8, r1 = r0 + 8;
      const bool v0 = r0 < PB, v1 = r1 < PB;
      const float b0 = bias ? __ldg(bias + c0 + 2 * tq) : 0.f, b1 = bias ? __ldg(bias + c0 + 2 * tq + 1) : 0.f;
      float d[4] = {b0, b1, b0, b1};
      tc_k<KD>(d, X, XL, W, ND, r0, r1, v0, v1, c0, gid, tq);
      if constexpr (KD2 > 0) tc_k<KD2>(d, X2, XL2, W2, ND, r0, r1, v0, v1, c0, gid, tq);
      if (v0) { Y[r0 * LDS + c0 + 2 * tq] = d[0]; Y[r0 * LDS + c0 + 2 * tq + 1] = d[1]; }
      if (v1) { Y[r1 * LDS + c0 + 2 * tq] = d[2]; Y[r1 * LDS + c0 + 2 * tq + 1] = d[3]; }
    }
  } else {
    mm<KD, KD2, ND>(X, XL, W, X2, XL2, W2, bias, Y);
  }
}

// member vectors in shared memory (HP each)
enum { V_mH, V_enc_b, V_encW0, V_encW1, V_encW2, V_encW3, V_dec_lin, NV_TOP };
#define VL(l, k) (NV_TOP + (l) * 5 + (k))                 // per layer: D, bv, bg, ln_g, ln_b
#define NV (NV_TOP + 5 * LP)
// per-patient constants (cbwd.py pack_q)
enum { Q_Ad = 0, Q_Bd = 16, Q_mu = 20, Q_stim, Q_stimt, Q_aou, Q_gam, Q_e0, Q_em, Q_dmask = 32, NQ = 48 };

__global__ void __launch_bounds__(NT, 1) cbwd_kernel(Args A, int B) {
  extern __shared__ float sm[];
  float* SU = sm;                                   // [LP][PB][HP] layer inputs
  float* SZV = SU + LP * PB * HP;                   // [LP][PB][HP]
  float* SZG = SZV + LP * PB * HP;                  // [LP][PB][HP]
  float* SLN = SZG + LP * PB * HP;                  // [LP][PB][2] (mean, 1 / std)
  float* SX = SLN + LP * PB * 2;                    // [PB][LDS] matmul input staging (TF32 hi)
  float* SXL = SX + PB * LDS;                       // [PB][LDS] TF32 lo parts
  float* SY = SXL + PB * LDS;                       // [PB][LDS] matmul output staging
  float* SV = SY + PB * LDS;                        // [NV][HP] member vectors
  float* SHD = SV + NV * HP;                        // [2][HDP] head_b1, head_dec
  float* SP5 = SHD + 2 * HDP;                       // [8][P5P] lamr lami Bur Bui Bwr Bwi Cor Coi
  float* SQ = SP5 + 8 * P5P;                        // [PB][NQ] patient constants
  const int tid = threadIdx.x, p = tid / LANES, lane = tid % LANES;
  const int g = blockIdx.y, bb = blockIdx.x * PB + p;
  const size_t gB = (size_t)g * B;

  // ---------------- one-time loads
  for (int i = tid; i < NV * HP; i += NT) {
    const int v = i / HP, h = i % HP;
    float x;
    if (v == V_mH) x = LD(mH, g * HP + h);
    else if (v == V_enc_b) x = LD(enc_b, g * HP + h);
    else if (v <= V_encW3) x = LD(encW, ((size_t)g * 4 + (v - V_encW0)) * HP + h);
    else if (v == V_dec_lin) x = LD(dec_lin, g * HP + h);
    else {
      const int l = (v - NV_TOP) / 5, kk = (v - NV_TOP) % 5;
      x = LD(L_vec, (((size_t)g * LP + l) * 5 + kk) * HP + h);
    }
    SV[i] = x;
  }
#if HEAD
  for (int i = tid; i < 2 * HDP; i += NT) SHD[i] = i < HDP ? LD(head_b1, g * HDP + i) : LD(head_dec, g * HDP + i - HDP);
#endif
  for (int i = tid; i < 8 * P5P; i += NT) SP5[i] = LD(p5, (size_t)g * 8 * P5P + i);
  for (int i = lane; i < NQ; i += LANES) SQ[p * NQ + i] = LD(q_c, (gB + bb) * NQ + i);
  __syncthreads();
  const float* q = SQ + p * NQ;
  const float nH = LD(sc, g * 8 + 0), ho = LD(sc, g * 8 + 1), u0 = LD(sc, g * 8 + 2), decb = LD(sc, g * 8 + 3),
              wlow = LD(sc, g * 8 + 4), rhou = LD(sc, g * 8 + 5);
  const float wc = 1.f / NSTEP;
  float glv[LP];
#pragma unroll
  for (int l = 0; l < LP; l++) glv[l] = LD(gl, g * LP + l);
#define SVv(v, j) SV[(v) * HP + lane + LANES * (j)]
#define STX(i, v) do { const float v_ = (v); const uint32_t hb_ = tf32(v_); SX[i] = __uint_as_float(hb_); SXL[i] = __uint_as_float(tf32(v_ - __uint_as_float(hb_))); } while (0)

#if GRADS
  float a_encW[4][CH], a_encb[CH], a_decl[CH], a_lng[LP][CH], a_lnb[LP][CH], a_hdec[CHD], a_p5[8][CP5];
  float a_decb = 0.f, a_u0 = 0.f;
#pragma unroll
  for (int j = 0; j < CH; j++) {
    a_encb[j] = 0.f; a_decl[j] = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) a_encW[i][j] = 0.f;
#pragma unroll
    for (int l = 0; l < LP; l++) { a_lng[l][j] = 0.f; a_lnb[l][j] = 0.f; }
  }
#pragma unroll
  for (int j = 0; j < CHD; j++) a_hdec[j] = 0.f;
#pragma unroll
  for (int v = 0; v < 8; v++)
#pragma unroll
    for (int j = 0; j < CP5; j++) a_p5[v][j] = 0.f;
#endif
  // S4D adjoint state and F (blocked adjoint scratch, owned per thread): zero
#pragma unroll
  for (int l = 0; l < LP; l++)
#pragma unroll
    for (int j = 0; j < CH; j++) {
      const int h = lane + LANES * j;
      for (int n = 0; n < NSP; n++)
        for (int c = 0; c < 2; c++) OUT(s_l)[(((((size_t)g * LP + l) * 2 + c) * B + bb) * NSP + n) * HP + h] = 0.f;
#pragma unroll
      for (int i = 0; i < TBLK; i++) OUT(s_F)[((((size_t)g * LP + l) * TBLK + i) * B + bb) * HP + h] = 0.f;
    }

  // ---------------- carry cotangent (of the carry after the current step; zero after the last step)
  float ctx[4] = {0.f, 0.f, 0.f, 0.f}, ctou = 0.f, ctup = 0.f, ctbuf[DPL];
#pragma unroll
  for (int j = 0; j < DPL; j++) ctbuf[j] = 0.f;
  float cthr[CP5], cthi[CP5];
#pragma unroll
  for (int j = 0; j < CP5; j++) { cthr[j] = 0.f; cthi[j] = 0.f; }

  const int NB = NSTEP / TBLK;
  for (int jb = NB - 1; jb >= 0; jb--) {
    for (int i = TBLK - 1; i >= 0; i--) {
      const int k = jb * TBLK + i;
      const float t = (float)k * (float)DT_;
      const size_t sk = (gB * NSTEP) + (size_t)k * B + bb;           // (g, k, b)
      // ======== tape the incoming carry adjoint (kstep.xla_backward: lam = cotangent of the carry after step k) ========
#if LAMOUT
      {
        float* lp = OUT(lam) + sk * S_TAPE;
        if (lane == 0) { lp[0] = ctx[0]; lp[1] = ctx[1]; lp[2] = ctx[2]; lp[3] = ctx[3]; lp[4] = ctou; lp[5] = ctup; }
#pragma unroll
        for (int j = 0; j < CP5; j++) {
          const int n = lane + LANES * j;
          if (n < P5P) { lp[6 + n] = cthr[j]; lp[6 + P5P + n] = cthi[j]; }
        }
#pragma unroll
        for (int j = 0; j < DPL; j++) if (lane + LANES * j < DPAD) lp[6 + 2 * P5P + lane + LANES * j] = ctbuf[j];
      }
#endif
      // ======== carry before this step (tape) and per-step inputs ========
      const float* cp = INP(c_tape) + sk * S_TAPE;
      float x[4];
      x[0] = __ldg(cp); x[1] = __ldg(cp + 1); x[2] = __ldg(cp + 2); x[3] = __ldg(cp + 3);
      const float ou = __ldg(cp + 4), up = __ldg(cp + 5);
      float bufc[DPL];
#pragma unroll
      for (int j = 0; j < DPL; j++) bufc[j] = lane + LANES * j < DPAD ? __ldg(cp + 6 + 2 * P5P + lane + LANES * j) : 0.f;
      const size_t si = ((size_t)g * 5 * NSTEP + k) * B + bb, sstr = (size_t)NSTEP * B;
      const float z0 = LD(s_in, si), tg = LD(s_in, si + sstr), sy = LD(s_in, si + 2 * sstr), r = LD(s_in, si + 3 * sstr), fr = LD(s_in, si + 4 * sstr);
      const float gg = q[Q_gam], e0 = q[Q_e0], em = q[Q_em];
      // ======== forward recompute: monitor, features ========
      const float ce = fmaxf(x[3], 0.f);
      const float logc50 = q[Q_mu] + ou + q[Q_stim] * (t > q[Q_stimt] ? 1.f : 0.f);
      float hpr[CP5], hpi[CP5];
#pragma unroll
      for (int j = 0; j < CP5; j++) {
        const int n = lane + LANES * j;
        hpr[j] = n < P5P ? __ldg(cp + 6 + n) * fr : 0.f;
        hpi[j] = n < P5P ? __ldg(cp + 6 + P5P + n) * fr : 0.f;
      }
      const float cg = powf(ce, gg);
      const float C = expf(logc50);
      const float Cg = powf(C, gg);
      const float den = cg + Cg;
      const float b = e0 - em * cg / den;
      const bool first = (k == 0);
      float dmask[DPL], ysum = 0.f;
#pragma unroll
      for (int j = 0; j < DPL; j++) {
        const int e = lane + LANES * j;
        const float v1 = BC(bufc[j], lane > 0 ? lane - 1 : 0), v2 = BC(bufc[j > 0 ? j - 1 : 0], LANES - 1);
        const float bufn = e >= DPAD ? 0.f : (first || e == 0) ? b : (lane > 0 ? v1 : v2);
        dmask[j] = e < DPAD ? q[Q_dmask + e] : 0.f;
        ysum += bufn * dmask[j];
      }
      const float y = AR(ysum) + sy * z0;
      const float s_raw = (E0_ - y) / EMAX_;
      const float s = fminf(1.f - 1e-3f, fmaxf(1e-3f, s_raw));
      const float ratio = s / (1.f - s);
      const float wpow = C50N_ * powf(ratio, 1.f / GAMMA_);
      const float w = fminf(wpow, 3.f * C50N_);
      const float f0 = (y - tg) / 50.f, f1 = up / (r * UMAX_), f2 = t / T_END_, f3 = w / (r * C50N_) - 1.f;
      float ul = 0.f;
#pragma unroll
      for (int j = 0; j < CP5; j++) {
        const int n = lane + LANES * j;
        if (n < P5P) ul += SP5[6 * P5P + n] * hpr[j] - SP5[7 * P5P + n] * hpi[j];
      }
      const float u_lin = u0 + AR(ul);
      float xx[CH];
#pragma unroll
      for (int j = 0; j < CH; j++)
        xx[j] = f0 * SVv(V_encW0, j) + f1 * SVv(V_encW1, j) + f2 * SVv(V_encW2, j) + f3 * SVv(V_encW3, j) + SVv(V_enc_b, j);
      // ======== forward recompute: residual layers (S4D outputs from the tape) ========
#pragma unroll
      for (int l = 0; l < LP; l++) {
        const float* ysk = INP(t_ys) + ((((size_t)g * LP + l) * NSTEP + k) * B + bb) * HP;
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const int h = lane + LANES * j;
          SU[(l * PB + p) * HP + h] = xx[j];
          float a, ga;
          gelu_d(__ldg(ysk + h) + SVv(VL(l, 0), j) * xx[j], a, ga);
          STX(p * LDS + h, a);
        }
        __syncthreads();
        const size_t wo = ((size_t)g * LP + l) * HP * HP;
        mmx<HP, 0, HP>(SX, SXL, INP(L_Wv) + wo, nullptr, nullptr, nullptr, INP(L_vec) + (((size_t)g * LP + l) * 5 + 1) * HP, SY);
        mmx<HP, 0, HP>(SX, SXL, INP(L_Wg) + wo, nullptr, nullptr, nullptr, INP(L_vec) + (((size_t)g * LP + l) * 5 + 2) * HP, SY + HP);
        __syncthreads();
        float v[CH], ps = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const int h = lane + LANES * j;
          const float zv = SY[p * LDS + h], zg = SY[p * LDS + HP + h];
          SZV[(l * PB + p) * HP + h] = zv; SZG[(l * PB + p) * HP + h] = zg;
          v[j] = xx[j] + zv * sigm(zg);
          ps += v[j] * SVv(V_mH, j);
        }
        const float m = AR(ps) / nH;
        float pv = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) { const float d = (v[j] - m) * SVv(V_mH, j); pv += d * d; }
        const float inv = 1.f / sqrtf(AR(pv) / nH + 1e-5f);
        if (lane == 0) { SLN[(l * PB + p) * 2] = m; SLN[(l * PB + p) * 2 + 1] = inv; }
        const float gl = glv[l];
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const float out = (v[j] - m) * SVv(V_mH, j) * inv * SVv(VL(l, 3), j) + SVv(VL(l, 4), j);
          xx[j] = gl * out + (1.f - gl) * xx[j];
        }
      }
      // ======== forward recompute: decoder / head ========
      float pol = 0.f;
#pragma unroll
      for (int j = 0; j < CH; j++) {
        const int h = lane + LANES * j;
        pol += xx[j] * SVv(V_dec_lin, j);
        STX(p * LDS + h, xx[j]);
#if GRADS
        OUT(t_xe)[((gB + bb) * NSTEP + k) * HP + h] = xx[j];
#endif
      }
      const float out_lin = AR(pol);
      float out_head = 0.f;
#if HEAD
      float hid[CHD], hgd[CHD];
      __syncthreads();
      mmx<HP, 0, HDP>(SX, SXL, INP(head_W1) + (size_t)g * HP * HDP, nullptr, nullptr, nullptr, INP(head_b1) + g * HDP, SY);
      __syncthreads();
      float poh = 0.f;
#pragma unroll
      for (int j = 0; j < CHD; j++) {
        const int n = lane + LANES * j;
        if (n < HDP) { gelu_d(SY[p * LDS + n], hid[j], hgd[j]); poh += hid[j] * SHD[HDP + n]; }
        else { hid[j] = 0.f; hgd[j] = 0.f; }
      }
      out_head = AR(poh);
#endif
      const float o = (1.f - ho) * out_lin + ho * out_head + decb;
      const float mean = r * (u_lin + 10.f * o);
      const float uapp = fminf(UMAX_, fmaxf(0.f, mean));
      const float un = uapp / r, wn = w / r;
      // ======================== backward ========================
      float d_b = wc * (2.f * (b - tg) / 100.f - wlow * 2.f * relu(40.f - b) / 100.f);
      float d_uapp = wc * rhou * 2.f * uapp / (UMAX_ * UMAX_);
      float d_x[4];
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        float s_ = 0.f;
#pragma unroll
        for (int ii = 0; ii < 4; ii++) s_ += q[Q_Ad + 4 * ii + jj] * ctx[ii];
        d_x[jj] = s_;
      }
#pragma unroll
      for (int ii = 0; ii < 4; ii++) d_uapp += q[Q_Bd + ii] * ctx[ii];
      float d_ou = q[Q_aou] * ctou;
      float d_hpr[CP5], d_hpi[CP5], pun = 0.f, pwn = 0.f;
#pragma unroll
      for (int j = 0; j < CP5; j++) {
        const int n = lane + LANES * j;
        if (n < P5P) {
          const float lr = SP5[n], li = SP5[P5P + n];
          d_hpr[j] = lr * cthr[j] + li * cthi[j];
          d_hpi[j] = -li * cthr[j] + lr * cthi[j];
          pun += SP5[2 * P5P + n] * cthr[j] + SP5[3 * P5P + n] * cthi[j];
          pwn += SP5[4 * P5P + n] * cthr[j] + SP5[5 * P5P + n] * cthi[j];
#if GRADS
          a_p5[0][j] += cthr[j] * hpr[j] + cthi[j] * hpi[j];
          a_p5[1][j] += -cthr[j] * hpi[j] + cthi[j] * hpr[j];
          a_p5[2][j] += cthr[j] * un; a_p5[3][j] += cthi[j] * un;
          a_p5[4][j] += cthr[j] * wn; a_p5[5][j] += cthi[j] * wn;
#endif
        } else { d_hpr[j] = 0.f; d_hpi[j] = 0.f; }
      }
      float d_w = AR(pwn) / r;
      d_uapp += ctup + AR(pun) / r;
      float d_bufn[DPL];
#pragma unroll
      for (int j = 0; j < DPL; j++) d_bufn[j] = ctbuf[j];
      const float d_m0 = d_uapp * d_clip(mean, 0.f, UMAX_) * r;
      const float d_ulin = d_m0;
      const float d_o = 10.f * d_m0;
      const float d_outlin = (1.f - ho) * d_o;
#if GRADS
      a_decb += d_o;
#endif
      // head backward
#if HEAD
      {
        const float d_outhead = ho * d_o;
#pragma unroll
        for (int j = 0; j < CHD; j++) {
          const int n = lane + LANES * j;
          if (n < HDP) {
            const float dp = d_outhead * SHD[HDP + n] * hgd[j];
#if GRADS
            a_hdec[j] += d_outhead * hid[j];
            OUT(t_dpre)[((gB + bb) * NSTEP + k) * HDP + n] = dp;
#endif
            STX(p * LDS + n, dp);
          }
        }
        __syncthreads();
        mmx<HDP, 0, HP>(SX, SXL, INP(W1T) + (size_t)g * HDP * HP, nullptr, nullptr, nullptr, nullptr, SY);
        __syncthreads();
      }
#endif
      float d_xx[CH];
#pragma unroll
      for (int j = 0; j < CH; j++) {
        const int h = lane + LANES * j;
#if HEAD
        d_xx[j] = SY[p * LDS + h] + d_outlin * SVv(V_dec_lin, j);
#else
        d_xx[j] = d_outlin * SVv(V_dec_lin, j);
        (void)h;
#endif
#if GRADS
        a_decl[j] += d_outlin * xx[j];
#endif
      }
      // physics output
#if GRADS
      a_u0 += d_ulin;
#endif
#pragma unroll
      for (int j = 0; j < CP5; j++) {
        const int n = lane + LANES * j;
        if (n < P5P) {
#if GRADS
          a_p5[6][j] += d_ulin * hpr[j]; a_p5[7][j] += -d_ulin * hpi[j];
#endif
          d_hpr[j] += d_ulin * SP5[6 * P5P + n];
          d_hpi[j] -= d_ulin * SP5[7 * P5P + n];
        }
      }
      // residual layers, last to first
#pragma unroll
      for (int l = LP - 1; l >= 0; l--) {
        const float gl = glv[l];
        const float m = SLN[(l * PB + p) * 2], inv = SLN[(l * PB + p) * 2 + 1];
        const float* ysk = INP(t_ys) + ((((size_t)g * LP + l) * NSTEP + k) * B + bb) * HP;
        const size_t to = ((((size_t)g * LP + l) * B + bb) * NSTEP + k) * HP;
        float* xk = OUT(t_x) + ((gB + bb) * NSTEP + k) * WX + l * 2 * HP;          // this step's (u, cy) of layer l
        float u[CH], dv[CH], tt[CH], d_u[CH], zvv[CH], sgv[CH], ptd = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const int h = lane + LANES * j;
          u[j] = SU[(l * PB + p) * HP + h];
          zvv[j] = SZV[(l * PB + p) * HP + h];
          sgv[j] = sigm(SZG[(l * PB + p) * HP + h]);
          const float mh = SVv(V_mH, j);
          dv[j] = (u[j] + zvv[j] * sgv[j] - m) * mh;
          const float d_out = gl * d_xx[j];
          d_u[j] = (1.f - gl) * d_xx[j];
#if GRADS
          a_lng[l][j] += d_out * dv[j] * inv;
          a_lnb[l][j] += d_out;
#endif
          tt[j] = d_out * SVv(VL(l, 3), j);
          ptd += tt[j] * dv[j];
        }
        const float s_td = AR(ptd);
        float d_dv[CH], pdd = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) {
          d_dv[j] = tt[j] * inv - inv * inv * inv * s_td * dv[j] / nH;
          pdd += d_dv[j] * SVv(V_mH, j);
        }
        const float s_dd = AR(pdd) / nH;
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const int h = lane + LANES * j;
          const float mh = SVv(V_mH, j);
          const float d_v = mh * (d_dv[j] - s_dd);
          d_u[j] += d_v;
          const float d_zv = d_v * sgv[j];
          const float d_zg = d_v * zvv[j] * sgv[j] * (1.f - sgv[j]);
#if GRADS
          OUT(t_dzv)[to + h] = d_zv; OUT(t_dzg)[to + h] = d_zg;
#endif
          STX(p * LDS + h, d_zv); STX(p * LDS + HP + h, d_zg);
        }
        __syncthreads();
        const size_t wo = ((size_t)g * LP + l) * HP * HP;
        mmx<HP, HP, HP>(SX, SXL, INP(L_WvT) + wo, SX + HP, SXL + HP, INP(L_WgT) + wo, nullptr, SY);
        __syncthreads();
        // S4D (blocked adjoint): cot(u) += e + K0 cy,  e = F_i + sum_{j > i in the block} K[j - i] cy_j
        const float* Kl = INP(b_K) + ((size_t)g * LP + l) * TBLK * HP;
        const float* cyb = OUT(t_x) + ((gB + bb) * NSTEP + jb * TBLK) * WX + l * 2 * HP + HP;
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const int h = lane + LANES * j;
          float a, ga;
          gelu_d(__ldg(ysk + h) + SVv(VL(l, 0), j) * u[j], a, ga);
          const float d_pa = SY[p * LDS + h] * ga;
          float ds = OUT(s_F)[((((size_t)g * LP + l) * TBLK + i) * B + bb) * HP + h];
#pragma unroll
          for (int jj = 1; jj < TBLK; jj++)
            if (i + jj < TBLK) ds += __ldg(Kl + jj * HP + h) * cyb[(i + jj) * WX + h];
          d_u[j] += d_pa * SVv(VL(l, 0), j) + ds + __ldg(Kl + h) * d_pa;
          xk[HP + h] = d_pa;
#if LAMOUT
          OUT(e)[((((size_t)g * LP + l) * NSTEP + k) * B + bb) * HP + h] = ds;
#endif
#if GRADS
          xk[h] = u[j];
          OUT(t_a)[to + h] = a;
#endif
          d_xx[j] = d_u[j];
        }
      }
      // encoder
      float pf[4] = {0.f, 0.f, 0.f, 0.f};
#if GRADS
      const float fv[4] = {f0, f1, f2, f3};
#endif
#pragma unroll
      for (int j = 0; j < CH; j++) {
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
#if GRADS
          a_encW[ii][j] += d_xx[j] * fv[ii];
#endif
          pf[ii] += d_xx[j] * SVv(V_encW0 + ii, j);
        }
#if GRADS
        a_encb[j] += d_xx[j];
#endif
      }
      const float d_f0 = AR(pf[0]), d_f1 = AR(pf[1]), d_f3 = AR(pf[3]);
      float d_y = d_f0 / 50.f;
      const float d_up = d_f1 / (r * UMAX_);
      d_w += d_f3 / (r * C50N_);
      // monitor
      const float d_wpow = d_w * d_min(wpow, 3.f * C50N_);
      const float d_ratio = d_wpow * C50N_ * (1.f / GAMMA_) * powf(ratio, 1.f / GAMMA_ - 1.f);
      const float d_s = d_ratio / ((1.f - s) * (1.f - s));
      d_y += d_s * d_clip(s_raw, 1e-3f, 1.f - 1e-3f) * (-1.f / EMAX_);
      float d_buf[DPL], sb = 0.f;
#pragma unroll
      for (int j = 0; j < DPL; j++) { d_bufn[j] += d_y * dmask[j]; sb += lane + LANES * j < DPAD ? d_bufn[j] : 0.f; }
      if (first) {
        d_b += AR(sb);
#pragma unroll
        for (int j = 0; j < DPL; j++) d_buf[j] = 0.f;
      } else {                                          // entry e of the carry's line feeds entry e + 1: d_buf[e] = d_bufn[e + 1]
        d_b += BC(d_bufn[0], 0);
#pragma unroll
        for (int j = 0; j < DPL; j++) {
          const int e = lane + LANES * j;
          const float dn = __shfl_down_sync(FULL, d_bufn[j], 1, LANES), nx = BC(d_bufn[j + 1 < DPL ? j + 1 : j], 0);
          d_buf[j] = e < DPAD - 1 ? (lane < LANES - 1 ? dn : nx) : 0.f;
        }
      }
      const float d_cg = -em * d_b * Cg / (den * den);
      const float d_Cg = em * d_b * cg / (den * den);
      const float d_ce = d_cg * gg * powf(ce, gg - 1.f);
      const float d_logc50 = d_Cg * gg * powf(C, gg - 1.f) * C;
      d_x[3] += d_ce * d_max(x[3], 0.f);
      d_ou += d_logc50;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) ctx[ii] = d_x[ii];
      ctou = d_ou; ctup = d_up;
#pragma unroll
      for (int j = 0; j < DPL; j++) ctbuf[j] = d_buf[j];
#pragma unroll
      for (int j = 0; j < CP5; j++) { cthr[j] = d_hpr[j] * fr; cthi[j] = d_hpi[j] * fr; }
    }
    // ======== block boundary: lam <- conj(dA^T) lam + 2 sum_i cy_i conj(C dA^(i+1)); F_i <- Re sum_n conj(lam) dB dA^(T-1-i) ========
#pragma unroll
    for (int l = 0; l < LP; l++)
#pragma unroll
      for (int j = 0; j < CH; j++) {
        const int h = lane + LANES * j;
        float cy[TBLK], F[TBLK];
        const float* cyb = OUT(t_x) + ((gB + bb) * NSTEP + jb * TBLK) * WX + l * 2 * HP + HP + h;
#pragma unroll
        for (int ii = 0; ii < TBLK; ii++) { cy[ii] = 2.f * cyb[ii * WX]; F[ii] = 0.f; }
        const size_t gl_ = (size_t)g * LP + l;
        for (int n = 0; n < NSP; n++) {
          const size_t so = (((gl_ * 2) * B + bb) * NSP + n) * HP + h, so2 = so + (size_t)B * NSP * HP;
          const size_t ao = ((gl_ * 2) * NSP + n) * HP + h;
          const float lr = OUT(s_l)[so], li = OUT(s_l)[so2];
          const float Ar = LD(b_AT, ao), Ai = LD(b_AT, ao + NSP * HP);
          float nr = Ar * lr + Ai * li, ni = Ar * li - Ai * lr;
#pragma unroll
          for (int ii = 0; ii < TBLK; ii++) {
            const size_t wo = (((gl_ * 4 + 0) * TBLK + ii) * NSP + n) * HP + h, ws = (size_t)TBLK * NSP * HP;
            nr += cy[ii] * LD(b_W, wo);
            ni -= cy[ii] * LD(b_W, wo + ws);
          }
          OUT(s_l)[so] = nr; OUT(s_l)[so2] = ni;
#pragma unroll
          for (int ii = 0; ii < TBLK; ii++) {
            const size_t wo = (((gl_ * 4 + 2) * TBLK + ii) * NSP + n) * HP + h, ws = (size_t)TBLK * NSP * HP;
            F[ii] += nr * LD(b_W, wo) + ni * LD(b_W, wo + ws);
          }
        }
#pragma unroll
        for (int ii = 0; ii < TBLK; ii++) OUT(s_F)[((gl_ * TBLK + ii) * B + bb) * HP + h] = F[ii];
      }
  }
#if GRADS
  // ---------------- per-patient gradients (packed, cbwd.py unpack_small)
  float* gs = OUT(g_sm) + (gB + bb) * NSM;
#pragma unroll
  for (int j = 0; j < CH; j++) {
    const int h = lane + LANES * j;
#pragma unroll
    for (int ii = 0; ii < 4; ii++) gs[ii * HP + h] = a_encW[ii][j];
    gs[4 * HP + h] = a_encb[j];
    gs[5 * HP + h] = a_decl[j];
#pragma unroll
    for (int l = 0; l < LP; l++) {
      gs[6 * HP + HDP + 8 * P5P + 2 + l * HP + h] = a_lng[l][j];
      gs[6 * HP + HDP + 8 * P5P + 2 + (LP + l) * HP + h] = a_lnb[l][j];
    }
  }
#pragma unroll
  for (int j = 0; j < CHD; j++) {
    const int n = lane + LANES * j;
    if (n < HDP) gs[6 * HP + n] = a_hdec[j];
  }
#pragma unroll
  for (int j = 0; j < CP5; j++) {
    const int n = lane + LANES * j;
    if (n < P5P)
#pragma unroll
      for (int v = 0; v < 8; v++) gs[6 * HP + HDP + v * P5P + n] = a_p5[v][j];
  }
  if (lane == 0) { gs[6 * HP + HDP + 8 * P5P] = a_decb; gs[6 * HP + HDP + 8 * P5P + 1] = a_u0; }
#endif
}

static size_t smem_bytes() {
  return sizeof(float) * (3 * LP * PB * HP + LP * PB * 2 + 3 * PB * LDS + NV * HP + 2 * HDP + 8 * P5P + PB * NQ);
}

static ffi::Error Impl(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets, int32_t B) {
  Args A;
  if ((int)args.size() != NIN || (int)rets.size() != NOUT) return ffi::Error::InvalidArgument("cbwd: wrong number of buffers");
  for (int i = 0; i < NIN; i++) {
    auto x = args.get<ffi::AnyBuffer>(i);
    if (x.has_error()) return ffi::Error::InvalidArgument("cbwd: bad input");
    A.in[i] = (const float*)x.value().untyped_data();
  }
  for (int i = 0; i < NOUT; i++) {
    auto y = rets.get<ffi::AnyBuffer>(i);
    if (y.has_error()) return ffi::Error::InvalidArgument("cbwd: bad output");
    A.out[i] = (float*)(*y.value()).untyped_data();
  }
  const int G = (int)(args.get<ffi::AnyBuffer>(I_sc).value().element_count() / 8);     // members (leading axis of every buffer)
  const size_t smem = smem_bytes();
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(cbwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr = true;
  }
  dim3 grid(B / PB, G);
  cbwd_kernel<<<grid, NT, smem, stream>>>(A, B);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CbwdHandler, Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets().Attr<int32_t>("B"));
