// cfwd.cu -- the forward sweep of kstep.py (kstep.xla_forward, tape and per-patient sums) as one CUDA kernel in the blocked layout:
// the S4D states live in global memory (L2) and are touched once per TBLK-step block; inside a block a layer's output is the state's part
// (2 Re sum_n C dA^(i+1) h, computed at the block start) plus the in-block taps K over the layer's inputs of the block (kept in registers), and
// at the block end the state is advanced over the block's inputs and the next block's state part is computed in the same pass (Horner's rule
// on dA, from dA, dB, C, dA^T). The per-step work
// (monitor, encoder, GELU / GLU / layer norm, head, physics, plant) is cbwd.cu's forward recompute: PB patients per thread block, LANES threads
// per patient, the GLU / head matmuls on tensor cores across the block's patients. Compile-time configuration: cfg.h (cbwd.py).
#include <cuda_runtime.h>
#include <stdint.h>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

#define NT (PB * LANES)
#define PD 2                                                   // block end: S4D modes loaded ahead of their use
#define CH (HP / LANES)
#define CHD ((HDP + LANES - 1) / LANES)
#define CP5 ((P5P + LANES - 1) / LANES)
#define FULL 0xffffffffu
#define PSZ(KD, ND) ((((ND) + 15) / 16) * 16 * (KD) * 2)            // floats of a packed tensor-core weight matrix
#define LDS (((2 * HP) > HDP ? (2 * HP) : HDP) + 4)                // (+ 4: rows on distinct banks)
#define S_TAPE (6 + 2 * P5P + DPAD)
#define DPL ((DPAD + LANES - 1) / LANES)                      // delay-line entries per lane: entry lane + LANES j


static_assert(PB % 8 == 0, "cfwd: the tensor-core tiles take 8 patients");
static_assert(NSP % PD == 0, "cfwd: the S4D modes come in groups of PD");

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
__device__ __forceinline__ float relu(float x) { return x > 0.f ? x : 0.f; }
// the special-function unit: 2^x and 1/x to about 2^-22 relative (no IEEE fix-up sequence; the arguments here are never denormal)
__device__ __forceinline__ float ex2a(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float rcpa(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
#define LOG2E 1.4426950408889634f
__device__ __forceinline__ float sigm(float x) { return rcpa(1.f + ex2a(-LOG2E * x)); }
__device__ __forceinline__ float gelu(float x) {                    // 0.5 x (1 + tanh u) = x / (1 + exp(-2u))
  const float c2 = 2.f * 0.7978845608028654f * LOG2E;
  return x * rcpa(1.f + ex2a(-c2 * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ uint32_t tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ void mma8(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Y[p][c] = sum_k X[p][k] W[k][c] + bias[c] on tensor cores (3xTF32), transposed: the weights are the 16-row A operand (W = hi + lo, packed
// by cbwd.py in fragment order: per 16-channel tile and 8-row k slice, each lane's 4 hi then 4 lo values), the block's patients the 8 columns
// of B (X = hi + lo staged in shared memory); one warp per 16 channels x 8 patients
template <int KD>
__device__ __forceinline__ void tc_k(float d[4], const float* X, const float* XL, const float* __restrict__ W, int mt, int p0, int ln, int gid, int tq) {
  const float4* f = reinterpret_cast<const float4*>(W) + ((size_t)mt * (KD / 8) * 32 + ln) * 2;
#pragma unroll 4
  for (int k0 = 0; k0 < KD; k0 += 8, f += 64) {                      // per k slice: 32 lanes x (4 hi, 4 lo)
    const float4 h = __ldg(f), lo = __ldg(f + 1);
    const uint32_t ah[4] = {__float_as_uint(h.x), __float_as_uint(h.y), __float_as_uint(h.z), __float_as_uint(h.w)};
    const uint32_t al[4] = {__float_as_uint(lo.x), __float_as_uint(lo.y), __float_as_uint(lo.z), __float_as_uint(lo.w)};
    const int xi = (p0 + gid) * LDS + k0 + tq;
    const uint32_t bh[2] = {__float_as_uint(X[xi]), __float_as_uint(X[xi + 4])};
    const uint32_t bl[2] = {__float_as_uint(XL[xi]), __float_as_uint(XL[xi + 4])};
    mma8(d, al, bh);
    mma8(d, ah, bl);
    mma8(d, ah, bh);
  }
}
// one or two products over the same X (the second when W2 != nullptr), their tiles spread over the block's warps
template <int KD, int ND>
__device__ __forceinline__ void mmx(const float* X, const float* XL, const float* __restrict__ W, const float* __restrict__ bias, float* Y,
                                    const float* __restrict__ W2 = nullptr, const float* __restrict__ bias2 = nullptr, float* Y2 = nullptr) {
  const int warp = threadIdx.x >> 5, ln = threadIdx.x & 31, gid = ln >> 2, tq = ln & 3;
  constexpr int MTL = (ND + 15) / 16, PT = PB / 8, NTI = MTL * PT;
  const int ntot = W2 ? 2 * NTI : NTI;
  for (int tile = warp; tile < ntot; tile += NT / 32) {
    const bool sec = tile >= NTI;
    const int tl = sec ? tile - NTI : tile, c0 = (tl % MTL) * 16, p0 = (tl / MTL) * 8;
    const float* bs = sec ? bias2 : bias;
    float* Yo = sec ? Y2 : Y;
    const bool up = ND % 16 == 0 || c0 + 8 < ND;
    const float b0 = bs ? __ldg(bs + c0 + gid) : 0.f, b1 = bs && up ? __ldg(bs + c0 + gid + 8) : 0.f;
    float d[4] = {b0, b0, b1, b1};
    tc_k<KD>(d, X, XL, sec ? W2 : W, tl % MTL, p0, ln, gid, tq);
    const int y0 = (p0 + 2 * tq) * LDS + c0 + gid;
    Yo[y0] = d[0]; Yo[y0 + LDS] = d[1];
    if (up) { Yo[y0 + 8] = d[2]; Yo[y0 + LDS + 8] = d[3]; }
  }
}

enum { V_mH, V_enc_b, V_encW0, V_encW1, V_encW2, V_encW3, V_dec_lin, NV_TOP };
#define VL(l, k) (NV_TOP + (l) * 6 + (k))                 // per layer: D, bv, bg, ln_g, ln_b, K0
#define NV (NV_TOP + 6 * LP)
enum { Q_Ad = 0, Q_Bd = 16, Q_mu = 20, Q_stim, Q_stimt, Q_aou, Q_gam, Q_e0, Q_em, Q_osc, Q_dmask = 32, NQ = 48 };

__global__ void __launch_bounds__(NT, 1) cfwd_kernel(Args A, int wtape) {
  constexpr int B = BN;                              // patients per member (compile time: constant offsets)
  extern __shared__ float sm[];
  float* SX = sm;                                   // [PB][LDS] matmul input staging (TF32 hi)
  float* SXL = SX + PB * LDS;                       // [PB][LDS] TF32 lo parts
  float* SY = SXL + PB * LDS;                       // [PB][LDS] matmul output staging
  float* SV = SY + PB * LDS;                        // [NV][HP] member vectors
  float* SHD = SV + NV * HP;                        // [2][HDP] head_b1, head_dec
  float* SP5 = SHD + 2 * HDP;                       // [8][P5P] lamr lami Bur Bui Bwr Bwi Cor Coi
  float* SK = SP5 + 8 * P5P;                        // [LP][TBLK][HP] in-block taps K[m], m = 0..T-1
  float* SQ = SK + LP * TBLK * HP;                  // [PB][NQ] patient constants
  const int tid = threadIdx.x, p = tid / LANES, lane = tid % LANES;
  const int g = blockIdx.y, bb = blockIdx.x * PB + p;
  const size_t gB = (size_t)g * B;

  for (int i = tid; i < NV * HP; i += NT) {
    const int v = i / HP, h = i % HP;
    float x;
    if (v == V_mH) x = LD(mH, g * HP + h);
    else if (v == V_enc_b) x = LD(enc_b, g * HP + h);
    else if (v <= V_encW3) x = LD(encW, ((size_t)g * 4 + (v - V_encW0)) * HP + h);
    else if (v == V_dec_lin) x = LD(dec_lin, g * HP + h);
    else {
      const int l = (v - NV_TOP) / 6, kk = (v - NV_TOP) % 6;
      x = kk < 5 ? LD(L_vec, (((size_t)g * LP + l) * 5 + kk) * HP + h) : LD(b_K, ((size_t)g * LP + l) * TBLK * HP + h);
    }
    SV[i] = x;
  }
#if HEAD
  for (int i = tid; i < 2 * HDP; i += NT) SHD[i] = i < HDP ? LD(head_b1, g * HDP + i) : LD(head_dec, g * HDP + i - HDP);
#endif
  for (int i = tid; i < 8 * P5P; i += NT) SP5[i] = LD(p5, (size_t)g * 8 * P5P + i);
  for (int i = tid; i < LP * TBLK * HP; i += NT) SK[i] = LD(b_K, (size_t)g * LP * TBLK * HP + i);
  for (int i = lane; i < NQ; i += LANES) SQ[p * NQ + i] = LD(q_c, (gB + bb) * NQ + i);
  __syncthreads();
  const float* q = SQ + p * NQ;
  const float nH = LD(sc, g * 8 + 0), ho = LD(sc, g * 8 + 1), u0 = LD(sc, g * 8 + 2), decb = LD(sc, g * 8 + 3),
              wlow = LD(sc, g * 8 + 4), rhou = LD(sc, g * 8 + 5);
  const float nHi = 1.f / nH;
  float glv[LP];
#pragma unroll
  for (int l = 0; l < LP; l++) glv[l] = LD(gl, g * LP + l);
#define SVv(v, j) SV[(v) * HP + lane + LANES * (j)]
#define STX(i, v) do { const float v_ = (v); const uint32_t hb_ = tf32(v_); SX[i] = __uint_as_float(hb_); SXL[i] = __uint_as_float(tf32(v_ - __uint_as_float(hb_))); } while (0)

  // carry
  float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f, ou = 0.f, up = 0.f, buf[DPL];
#pragma unroll
  for (int j = 0; j < DPL; j++) buf[j] = lane + LANES * j < DPAD ? E0_ : 0.f;
  float hpr[CP5], hpi[CP5];
#pragma unroll
  for (int j = 0; j < CP5; j++) { hpr[j] = 0.f; hpi[j] = 0.f; }
  float s_cost = 0.f, s_se = 0.f, s_v = 0.f;
  // blocked S4D: the state's part of the outputs of the current block, and the block's layer inputs
  float base[LP][CH][TBLK], ub[LP][CH][TBLK];
#pragma unroll
  for (int l = 0; l < LP; l++)
#pragma unroll
    for (int j = 0; j < CH; j++)
#pragma unroll
      for (int i = 0; i < TBLK; i++) base[l][j][i] = 0.f;
#pragma unroll
  for (int l = 0; l < LP; l++)
#pragma unroll
    for (int j = 0; j < CH; j++) {
      const int h = lane + LANES * j;
      for (int n = 0; n < NSP; n++)
        for (int c = 0; c < 2; c++) OUT(s_h)[(((((size_t)g * LP + l) * 2 + c) * B + bb) * NSP + n) * HP + h] = 0.f;
    }

  const int NB = NSTEP / TBLK;
  for (int jb = 0; jb < NB; jb++) {
#pragma unroll
    for (int i = 0; i < TBLK; i++) {
      const int k = jb * TBLK + i;
      const float t = (float)k * (float)DT_;
      const size_t sk = gB * NSTEP + (size_t)k * B + bb;
      if (wtape) {
        float* tp = OUT(tape) + sk * S_TAPE;
        if (lane == 0) { tp[0] = x0; tp[1] = x1; tp[2] = x2; tp[3] = x3; tp[4] = ou; tp[5] = up; }
#pragma unroll
        for (int j = 0; j < CP5; j++) {
          const int n = lane + LANES * j;
          if (n < P5P) { tp[6 + n] = hpr[j]; tp[6 + P5P + n] = hpi[j]; }
        }
#pragma unroll
        for (int j = 0; j < DPL; j++) if (lane + LANES * j < DPAD) tp[6 + 2 * P5P + lane + LANES * j] = buf[j];
      }
      const size_t si = ((size_t)g * 6 * NSTEP + k) * B + bb, sstr = (size_t)NSTEP * B;
      const float z0 = LD(s_in, si), tg = LD(s_in, si + sstr), sy = LD(s_in, si + 2 * sstr), r = LD(s_in, si + 3 * sstr),
                  fr = LD(s_in, si + 4 * sstr), z1 = LD(s_in, si + 5 * sstr);
      // monitor, features
      const float ce = fmaxf(x3, 0.f);
      const float logc50 = q[Q_mu] + ou + q[Q_stim] * (t > q[Q_stimt] ? 1.f : 0.f);
      const float cg = powf(ce, q[Q_gam]);
      const float Cg = expf(q[Q_gam] * logc50);                         // (e^logc50)^gamma
      const float b = q[Q_e0] - q[Q_em] * cg / (cg + Cg);
      float ysum = 0.f;
      {
        float nb[DPL];                                  // shift by one entry: entry e takes e - 1 (lane - 1, or the last lane of the previous j)
#pragma unroll
        for (int j = 0; j < DPL; j++) {
          const int e = lane + LANES * j;
          const float v1 = BC(buf[j], lane > 0 ? lane - 1 : 0), v2 = BC(buf[j > 0 ? j - 1 : 0], LANES - 1);
          nb[j] = e >= DPAD ? 0.f : (k == 0 || e == 0) ? b : (lane > 0 ? v1 : v2);
        }
#pragma unroll
        for (int j = 0; j < DPL; j++) {
          buf[j] = nb[j];
          if (lane + LANES * j < DPAD) ysum += buf[j] * q[Q_dmask + lane + LANES * j];
        }
      }
      const float y = AR(ysum) + sy * z0;
      const float s = fminf(1.f - 1e-3f, fmaxf(1e-3f, (E0_ - y) * (1.f / EMAX_)));
      const float w = fminf(C50N_ * powf(s / (1.f - s), 1.f / GAMMA_), 3.f * C50N_);
      const float ri = __frcp_rn(r);
      const float f0 = (y - tg) * 0.02f, f1 = up * ri * (1.f / UMAX_), f2 = t * (1.f / T_END_), f3 = w * ri * (1.f / C50N_) - 1.f;
      float hqr[CP5], hqi[CP5], ul = 0.f;
#pragma unroll
      for (int j = 0; j < CP5; j++) {
        const int n = lane + LANES * j;
        hqr[j] = hpr[j] * fr; hqi[j] = hpi[j] * fr;
        if (n < P5P) ul += SP5[6 * P5P + n] * hqr[j] - SP5[7 * P5P + n] * hqi[j];
      }
      const float u_lin = u0 + AR(ul);
      float xx[CH];
#pragma unroll
      for (int j = 0; j < CH; j++)
        xx[j] = f0 * SVv(V_encW0, j) + f1 * SVv(V_encW1, j) + f2 * SVv(V_encW2, j) + f3 * SVv(V_encW3, j) + SVv(V_enc_b, j);
      // layers
#pragma unroll
      for (int l = 0; l < LP; l++) {
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const int h = lane + LANES * j;
          ub[l][j][i] = xx[j];
          float ys = base[l][j][i];
#pragma unroll
          for (int i2 = 0; i2 <= i; i2++) ys += SK[(l * TBLK + (i - i2)) * HP + h] * ub[l][j][i2];
          if (wtape) OUT(ys)[((((size_t)g * LP + l) * NSTEP + k) * B + bb) * HP + h] = ys;
          STX(p * LDS + h, gelu(ys + SVv(VL(l, 0), j) * xx[j]));
        }
        __syncthreads();
        const size_t wo = ((size_t)g * LP + l) * PSZ(HP, HP);
        mmx<HP, HP>(SX, SXL, INP(L_Wv) + wo, INP(L_vec) + (((size_t)g * LP + l) * 5 + 1) * HP, SY,
                    INP(L_Wg) + wo, INP(L_vec) + (((size_t)g * LP + l) * 5 + 2) * HP, SY + HP);
        __syncthreads();
        float v[CH], ps = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const int h = lane + LANES * j;
          v[j] = xx[j] + SY[p * LDS + h] * sigm(SY[p * LDS + HP + h]);
          ps += v[j] * SVv(V_mH, j);
        }
        const float m = AR(ps) * nHi;
        float pv = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) { const float d = (v[j] - m) * SVv(V_mH, j); pv += d * d; }
        const float inv = rsqrtf(AR(pv) * nHi + 1e-5f);
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const float out = (v[j] - m) * SVv(V_mH, j) * inv * SVv(VL(l, 3), j) + SVv(VL(l, 4), j);
          xx[j] = glv[l] * out + (1.f - glv[l]) * xx[j];
        }
      }
      // head
      float pol = 0.f;
#pragma unroll
      for (int j = 0; j < CH; j++) { pol += xx[j] * SVv(V_dec_lin, j); STX(p * LDS + lane + LANES * j, xx[j]); }
      const float out_lin = AR(pol);
      float out_head = 0.f;
#if HEAD
      __syncthreads();
      mmx<HP, HDP>(SX, SXL, INP(head_W1) + (size_t)g * PSZ(HP, HDP), INP(head_b1) + g * HDP, SY);
      __syncthreads();
      float poh = 0.f;
#pragma unroll
      for (int j = 0; j < CHD; j++) {
        const int n = lane + LANES * j;
        if (n < HDP) poh += gelu(SY[p * LDS + n]) * SHD[HDP + n];
      }
      out_head = AR(poh);
#endif
      const float o = (1.f - ho) * out_lin + ho * out_head + decb;
      const float uu = fminf(UMAX_, fmaxf(0.f, r * (u_lin + 10.f * o)));
      const float un = uu * ri, wn = w * ri;
#pragma unroll
      for (int j = 0; j < CP5; j++) {
        const int n = lane + LANES * j;
        if (n < P5P) {
          const float lr = SP5[n], li = SP5[P5P + n];
          hpr[j] = lr * hqr[j] - li * hqi[j] + SP5[2 * P5P + n] * un + SP5[4 * P5P + n] * wn;
          hpi[j] = lr * hqi[j] + li * hqr[j] + SP5[3 * P5P + n] * un + SP5[5 * P5P + n] * wn;
        }
      }
      const float lo = relu(40.f - b);
      s_cost += (b - tg) * (b - tg) * 0.01f + wlow * lo * lo * 0.01f + rhou * (uu * (1.f / UMAX_)) * (uu * (1.f / UMAX_));
      s_se += (b - tg) * (b - tg);
      s_v += lo * lo;
      const float n0 = q[0] * x0 + q[1] * x1 + q[2] * x2 + q[3] * x3 + q[Q_Bd] * uu;
      const float n1 = q[4] * x0 + q[5] * x1 + q[6] * x2 + q[7] * x3 + q[Q_Bd + 1] * uu;
      const float n2 = q[8] * x0 + q[9] * x1 + q[10] * x2 + q[11] * x3 + q[Q_Bd + 2] * uu;
      const float n3 = q[12] * x0 + q[13] * x1 + q[14] * x2 + q[15] * x3 + q[Q_Bd + 3] * uu;
      x0 = n0; x1 = n1; x2 = n2; x3 = n3;
      ou = q[Q_aou] * ou + q[Q_osc] * z1;
      up = uu;
    }
    // block end: h <- dA^T h + dB P with P = sum_j u_j dA^(T-1-j) (Horner's rule on the inputs alone), and the next block's state part
    // base_i = Re (w dA^i), w = 2 C dA h (w <- w dA per step): 9 T + 12 operations per mode instead of 13 T
#pragma unroll
    for (int l = 0; l < LP; l++)
#pragma unroll
      for (int j = 0; j < CH; j++) {
        const int h = lane + LANES * j;
        const size_t gl_ = (size_t)g * LP + l;
#pragma unroll
        for (int i = 0; i < TBLK; i++) base[l][j][i] = 0.f;
        const size_t so0 = (((gl_ * 2) * B + bb) * NSP) * HP + h, s2 = (size_t)B * NSP * HP, ao0 = (gl_ * 6 * NSP) * HP + h, as = (size_t)NSP * HP;
        const size_t to0 = (gl_ * 2 * NSP) * HP + h;
        float c_[PD][10];                                              // modes n .. n + PD - 1: dA, dB, C, state, dA^T, loaded PD modes ahead
#pragma unroll
        for (int d = 0; d < PD; d++) {
#pragma unroll
          for (int k = 0; k < 6; k++) c_[d][k] = LD(b_S4, ao0 + k * as + d * HP);
          c_[d][6] = OUT(s_h)[so0 + d * HP]; c_[d][7] = OUT(s_h)[so0 + s2 + d * HP];
          c_[d][8] = LD(b_AT, to0 + d * HP); c_[d][9] = LD(b_AT, to0 + as + d * HP);
        }
#pragma unroll 2
        for (int n0 = 0; n0 < NSP; n0 += PD)
#pragma unroll
        for (int d = 0; d < PD; d++) {
          const int n = n0 + d;
          const size_t so = so0 + (size_t)n * HP, so2 = so + s2;
          const float Ar = c_[d][0], Ai = c_[d][1], Br = c_[d][2], Bi = c_[d][3], Cr = c_[d][4], Ci = c_[d][5];
          const float sr = c_[d][6], si = c_[d][7], Tr = c_[d][8], Ti = c_[d][9];
          if (n + PD < NSP) {
#pragma unroll
            for (int k = 0; k < 6; k++) c_[d][k] = LD(b_S4, ao0 + k * as + (n + PD) * HP);
            c_[d][6] = OUT(s_h)[so + PD * HP]; c_[d][7] = OUT(s_h)[so2 + PD * HP];
            c_[d][8] = LD(b_AT, to0 + (n + PD) * HP); c_[d][9] = LD(b_AT, to0 + as + (n + PD) * HP);
          }
          float pr = ub[l][j][0], pi = 0.f;
#pragma unroll
          for (int i = 1; i < TBLK; i++) {
            const float tr = fmaf(pr, Ar, fmaf(-pi, Ai, ub[l][j][i]));
            pi = fmaf(pr, Ai, pi * Ar);
            pr = tr;
          }
          const float nr = Tr * sr - Ti * si + Br * pr - Bi * pi, ni = Tr * si + Ti * sr + Br * pi + Bi * pr;
          OUT(s_h)[so] = nr; OUT(s_h)[so2] = ni;
          float zr = 2.f * (Cr * nr - Ci * ni), zi = 2.f * (Cr * ni + Ci * nr);
#pragma unroll
          for (int i = 0; i < TBLK; i++) {
            const float tr = Ar * zr - Ai * zi;
            zi = Ar * zi + Ai * zr;
            zr = tr;
            base[l][j][i] += zr;
          }
        }
      }
  }
  if (lane == 0) { OUT(sums)[gB * 3 + bb] = s_cost; OUT(sums)[gB * 3 + B + bb] = s_se; OUT(sums)[gB * 3 + 2 * B + bb] = s_v; }
}

static size_t smem_bytes() { return sizeof(float) * (3 * PB * LDS + NV * HP + 2 * HDP + 8 * P5P + LP * TBLK * HP + PB * NQ); }

static ffi::Error Impl(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets, int32_t B, int32_t wtape) {
  Args A;
  if ((int)args.size() != NIN || (int)rets.size() != NOUT) return ffi::Error::InvalidArgument("cfwd: wrong number of buffers");
  for (int i = 0; i < NIN; i++) {
    auto x = args.get<ffi::AnyBuffer>(i);
    if (x.has_error()) return ffi::Error::InvalidArgument("cfwd: bad input");
    A.in[i] = (const float*)x.value().untyped_data();
  }
  for (int i = 0; i < NOUT; i++) {
    auto y = rets.get<ffi::AnyBuffer>(i);
    if (y.has_error()) return ffi::Error::InvalidArgument("cfwd: bad output");
    A.out[i] = (float*)(*y.value()).untyped_data();
  }
  const int G = (int)(args.get<ffi::AnyBuffer>(I_sc).value().element_count() / 8);
  const size_t smem = smem_bytes();
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(cfwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr = true;
  }
  dim3 grid(B / PB, G);
  if (B != BN) return ffi::Error::InvalidArgument("cfwd: built for another batch size");
  cfwd_kernel<<<grid, NT, smem, stream>>>(A, wtape);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CfwdHandler, Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets().Attr<int32_t>("B").Attr<int32_t>("wtape"));
