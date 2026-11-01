// cfwd.cu -- the forward sweep of kstep.py (kstep.xla_forward, tape and per-patient sums) as one CUDA kernel in the blocked layout:
// the S4D states live in global memory (L2) and are touched once per TBLK-step block; inside a block a layer's output is the state's part
// (precomputed at the block start, Wb = C dA^(i+1)) plus the in-block taps K over the layer's inputs of the block (kept in registers), and at
// the block end the state is advanced with Wu = dB dA^(T-1-j) and the next block's state part is computed in the same pass. The per-step work
// (monitor, encoder, GELU / GLU / layer norm, head, physics, plant) is cbwd.cu's forward recompute: PB patients per thread block, LANES threads
// per patient, the GLU / head matmuls on tensor cores across the block's patients. Compile-time configuration: cfg.h (cbwd.py).
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
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float gelu(float x) {
  const float c = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ uint32_t tf32(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r; }
__device__ __forceinline__ void mma8(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// Y[p][n] = sum_k X[p][k] W[k][n] + bias[n] on tensor cores (3xTF32); X = hi + lo staged in shared memory, rows = patients
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
template <int KD, int ND>
__device__ __forceinline__ void mmx(const float* X, const float* XL, const float* __restrict__ W, const float* __restrict__ bias, float* Y) {
  const int warp = threadIdx.x >> 5, ln = threadIdx.x & 31, gid = ln >> 2, tq = ln & 3;
  constexpr int NTL = ND / 8;
  constexpr int MT = (PB + 15) / 16;
  for (int tile = warp; tile < MT * NTL; tile += NT / 32) {
    const int r0 = (tile / NTL) * 16 + gid, c0 = (tile % NTL) * 8, r1 = r0 + 8;
    const bool v0 = r0 < PB, v1 = r1 < PB;
    const float b0 = bias ? __ldg(bias + c0 + 2 * tq) : 0.f, b1 = bias ? __ldg(bias + c0 + 2 * tq + 1) : 0.f;
    float d[4] = {b0, b1, b0, b1};
    tc_k<KD>(d, X, XL, W, ND, r0, r1, v0, v1, c0, gid, tq);
    if (v0) { Y[r0 * LDS + c0 + 2 * tq] = d[0]; Y[r0 * LDS + c0 + 2 * tq + 1] = d[1]; }
    if (v1) { Y[r1 * LDS + c0 + 2 * tq] = d[2]; Y[r1 * LDS + c0 + 2 * tq + 1] = d[3]; }
  }
}

enum { V_mH, V_enc_b, V_encW0, V_encW1, V_encW2, V_encW3, V_dec_lin, NV_TOP };
#define VL(l, k) (NV_TOP + (l) * 6 + (k))                 // per layer: D, bv, bg, ln_g, ln_b, K0
#define NV (NV_TOP + 6 * LP)
enum { Q_Ad = 0, Q_Bd = 16, Q_mu = 20, Q_stim, Q_stimt, Q_aou, Q_gam, Q_e0, Q_em, Q_osc, Q_dmask = 32, NQ = 48 };

__global__ void __launch_bounds__(NT, 1) cfwd_kernel(Args A, int B, int wtape) {
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
      const float Cg = powf(expf(logc50), q[Q_gam]);
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
      const float s = fminf(1.f - 1e-3f, fmaxf(1e-3f, (E0_ - y) / EMAX_));
      const float w = fminf(C50N_ * powf(s / (1.f - s), 1.f / GAMMA_), 3.f * C50N_);
      const float f0 = (y - tg) / 50.f, f1 = up / (r * UMAX_), f2 = t / T_END_, f3 = w / (r * C50N_) - 1.f;
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
        const size_t wo = ((size_t)g * LP + l) * HP * HP;
        mmx<HP, HP>(SX, SXL, INP(L_Wv) + wo, INP(L_vec) + (((size_t)g * LP + l) * 5 + 1) * HP, SY);
        mmx<HP, HP>(SX, SXL, INP(L_Wg) + wo, INP(L_vec) + (((size_t)g * LP + l) * 5 + 2) * HP, SY + HP);
        __syncthreads();
        float v[CH], ps = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) {
          const int h = lane + LANES * j;
          v[j] = xx[j] + SY[p * LDS + h] * sigm(SY[p * LDS + HP + h]);
          ps += v[j] * SVv(V_mH, j);
        }
        const float m = AR(ps) / nH;
        float pv = 0.f;
#pragma unroll
        for (int j = 0; j < CH; j++) { const float d = (v[j] - m) * SVv(V_mH, j); pv += d * d; }
        const float inv = 1.f / sqrtf(AR(pv) / nH + 1e-5f);
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
      mmx<HP, HDP>(SX, SXL, INP(head_W1) + (size_t)g * HP * HDP, INP(head_b1) + g * HDP, SY);
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
      const float un = uu / r, wn = w / r;
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
      s_cost += (b - tg) * (b - tg) / 100.f + wlow * lo * lo / 100.f + rhou * (uu / UMAX_) * (uu / UMAX_);
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
    // block end: h <- dA^T h + sum_j Wu_j u_j (Wu_j = dB dA^(T-1-j)), and the next block's state part base_i = 2 Re sum_n C dA^(i+1) h
#pragma unroll
    for (int l = 0; l < LP; l++)
#pragma unroll
      for (int j = 0; j < CH; j++) {
        const int h = lane + LANES * j;
        const size_t gl_ = (size_t)g * LP + l;
#pragma unroll
        for (int i = 0; i < TBLK; i++) base[l][j][i] = 0.f;
        for (int n = 0; n < NSP; n++) {
          const size_t so = (((gl_ * 2) * B + bb) * NSP + n) * HP + h, so2 = so + (size_t)B * NSP * HP;
          const size_t ao = ((gl_ * 2) * NSP + n) * HP + h;
          const float hr = OUT(s_h)[so], hi = OUT(s_h)[so2];
          const float Ar = LD(b_AT, ao), Ai = LD(b_AT, ao + NSP * HP);
          float nr = Ar * hr - Ai * hi, ni = Ar * hi + Ai * hr;
          const size_t ws = (size_t)TBLK * NSP * HP;
#pragma unroll
          for (int i = 0; i < TBLK; i++) {
            const size_t wo = (((gl_ * 4 + 2) * TBLK + i) * NSP + n) * HP + h;      // Wu (re, im)
            nr += LD(b_W, wo) * ub[l][j][i];
            ni += LD(b_W, wo + ws) * ub[l][j][i];
          }
          OUT(s_h)[so] = nr; OUT(s_h)[so2] = ni;
#pragma unroll
          for (int i = 0; i < TBLK; i++) {
            const size_t wo = (((gl_ * 4 + 0) * TBLK + i) * NSP + n) * HP + h;      // Wb (re, im)
            base[l][j][i] += 2.f * (LD(b_W, wo) * nr - LD(b_W, wo + ws) * ni);
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
  cfwd_kernel<<<grid, NT, smem, stream>>>(A, B, wtape);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CfwdHandler, Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets().Attr<int32_t>("B").Attr<int32_t>("wtape"));
