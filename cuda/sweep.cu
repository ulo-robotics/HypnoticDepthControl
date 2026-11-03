// CUDA implementation of the two sweeps of kstep.py (forward with tapes, adjoint backward), called from JAX as XLA FFI custom calls.
// One warp per patient; lane j owns channels j, j + 32, ... (S4D states / adjoints in registers) and physics mode j. The sizes are compile-time
// constants (-DH=.. -DNS=.. -DNL=.. -DHD=.. -DPP=.. -DWPB=.. -DSTAGE=0|1): kcuda.py builds one library per shape. The math is kstep.step; the backward step
// is its VJP with respect to the small carry and the S4D outputs, written out by hand (checked against the XLA sweeps by kcuda.selftest).
// Layouts (kcuda.py packs / unpacks them):
//   kp   flat parameters: sc[4] | lamr lami Bur Bui Bwr Bwi Cor Coi [PP each] | enc0..enc3 enc_b [H each] | dec [HD or H] |
//        (HD > 0) hW [H*HD] hWT [HD*H] hb [HD] | per layer: Ar Ai Br Bi Cr Ci [H*NS] K0 D [H] Wv WvT Wg WgT [H*H] bv bg g beta [H]
//   q    (B, 48): mu a_ou osc stim gamma e0 emax | dmask[16] | Ad[16] (row major) | Bd[4] | stim_t | stim2 stim_t2 stim3 stim_t3
//   st   (7, N, B): z0 z1 tg sy r rho w   (w: the cost weight of the step, N [k < len] / len)
//   tape (N, B, S), S = 6 + 2 PP + 16: x0 x1 x2 x3 ou up | hpr[PP] | hpi[PP] | buf[16]   (forward: carry before step k; backward: adjoint of the carry after step k)
//   ys / e  (NL, N, B, H);  sums (3, B): cost, (b - tg)^2, relu(40 - b)^2 summed over the episode
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#ifndef H
#error "compile with -DH=.. -DNS=.. -DNL=.. -DHD=.. -DPP=.. -DWPB=.."
#endif
#ifndef NSTEP
#define NSTEP 360
#endif
constexpr int DP = 16, NQ = 48;
constexpr float DT = 5.0f / 60.0f, T_END = 30.0f, UMAX = 100.0f, E0N = 100.0f, EMAXN = 100.0f, GAMN = 2.0f, C50N = 4.5f;
constexpr int CH = (H + 31) / 32;                       // channels per lane
constexpr int HDV = HD > 0 ? HD : H;                    // length of the decoder vector
constexpr int CHD = (HDV + 31) / 32;
constexpr int S = 6 + 2 * PP + DP;
constexpr int VMAX = (H > HDV ? H : HDV);
// weight staging: the warps of a block (all at the same step and layer) copy the matrices in use into shared memory once and share them
constexpr int WS_N = (2 * H * H > H * HD ? 2 * H * H : H * HD);
#ifndef STAGE
#define STAGE 0
#endif
constexpr bool WSMEM = STAGE && WS_N * 4 <= 40 * 1024;          // tuned: pays off with many warps per block and large H
// parameter offsets
constexpr int O_SC = 0, O_LAMR = 4, O_LAMI = O_LAMR + PP, O_BUR = O_LAMI + PP, O_BUI = O_BUR + PP, O_BWR = O_BUI + PP, O_BWI = O_BWR + PP,
              O_COR = O_BWI + PP, O_COI = O_COR + PP, O_ENC = O_COI + PP, O_ENCB = O_ENC + 4 * H, O_DEC = O_ENCB + H, O_HW = O_DEC + HDV,
              O_HWT = O_HW + H * HD, O_HB = O_HWT + HD * H, O_LAY = O_HB + HD;
constexpr int L_AR = 0, L_AI = H * NS, L_BR = 2 * H * NS, L_BI = 3 * H * NS, L_CR = 4 * H * NS, L_CI = 5 * H * NS, L_K0 = 6 * H * NS, L_D = L_K0 + H,
              L_WV = L_D + H, L_WVT = L_WV + H * H, L_WG = L_WVT + H * H, L_WGT = L_WG + H * H, L_BV = L_WGT + H * H, L_BG = L_BV + H,
              L_G = L_BG + H, L_BETA = L_G + H, L_STRIDE = L_BETA + H;

__device__ __forceinline__ float ld(const float* p) { return __ldg(p); }

// copy n1 floats from g1 (and n2 from g2) into the block's staging buffer; returns the base to read the matrices from (global without WSMEM)
__device__ __forceinline__ const float* stage(float* ws, const float* g1, int n1, const float* g2 = nullptr, int n2 = 0) {
  if (!WSMEM) return nullptr;
  __syncthreads();
  for (int i = threadIdx.x; i < n1; i += blockDim.x) ws[i] = g1[i];
  for (int i = threadIdx.x; i < n2; i += blockDim.x) ws[n1 + i] = g2[i];
  __syncthreads();
  return ws;
}
__device__ __forceinline__ float rd(const float* p) { return WSMEM ? *p : __ldg(p); }
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float gelu(float x) {
  const float k = 0.7978845608028654f;
  return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float gelu_d(float x) {
  const float k = 0.7978845608028654f;
  float t = tanhf(k * (x + 0.044715f * x * x * x));
  return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * k * (1.0f + 3.0f * 0.044715f * x * x);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float relu(float x) { return x > 0.0f ? x : 0.0f; }

struct Patient {      // per-patient constants (warp-uniform)
  float mu, a_ou, osc, stim, gam, e0, emax, dmask[DP], A[16], Bd[4], stim_t, stim2, stim_t2, stim3, stim_t3;
  __device__ void load(const float* q, int b) {
    const float* r = q + (size_t)b * NQ;
    mu = r[0]; a_ou = r[1]; osc = r[2]; stim = r[3]; gam = r[4]; e0 = r[5]; emax = r[6];
#pragma unroll
    for (int j = 0; j < DP; j++) dmask[j] = r[7 + j];
#pragma unroll
    for (int j = 0; j < 16; j++) A[j] = r[23 + j];
#pragma unroll
    for (int j = 0; j < 4; j++) Bd[j] = r[39 + j];
    stim_t = r[43]; stim2 = r[44]; stim_t2 = r[45]; stim3 = r[46]; stim_t3 = r[47];
  }
};

struct Pre {          // pre stage of a step (warp-uniform scalars)
  float ce, cg, eg, b, y, s0, s, w0, w, f0, f1, f2, f3;
};

// monitor, delay line, inverse Hill, features. buf is updated in place (to buf')
__device__ __forceinline__ void pre_stage(const Patient& P, int k, float x3, float ou, float up, float* buf, float z0, float sy, float tg, float r,
                                          Pre& o) {
  float t = k * DT;
  o.ce = fmaxf(x3, 0.0f);
  float l50 = P.mu + ou + P.stim * (t > P.stim_t ? 1.0f : 0.0f) + P.stim2 * (t > P.stim_t2 ? 1.0f : 0.0f) + P.stim3 * (t > P.stim_t3 ? 1.0f : 0.0f);
  o.cg = powf(o.ce, P.gam);
  o.eg = powf(expf(l50), P.gam);
  o.b = P.e0 - P.emax * o.cg / (o.cg + o.eg);
  if (k == 0) {
#pragma unroll
    for (int j = 0; j < DP; j++) buf[j] = o.b;
  } else {
#pragma unroll
    for (int j = DP - 1; j > 0; j--) buf[j] = buf[j - 1];
    buf[0] = o.b;
  }
  float bd = 0.0f;
#pragma unroll
  for (int j = 0; j < DP; j++) bd += buf[j] * P.dmask[j];
  o.y = bd + sy * z0;
  o.s0 = (E0N - o.y) / EMAXN;
  o.s = fminf(fmaxf(o.s0, 1e-3f), 1.0f - 1e-3f);
  o.w0 = C50N * powf(o.s / (1.0f - o.s), 1.0f / GAMN);
  o.w = fminf(o.w0, 3.0f * C50N);
  o.f0 = (o.y - tg) / 50.0f;
  o.f1 = up / (r * UMAX);
  o.f2 = 0.0f;
  o.f3 = o.w / (r * C50N) - 1.0f;
}

// a (owned channels) -> zv, zg (owned channels): z = a @ W + bias. sm: this warp's broadcast buffer
__device__ __forceinline__ void glu_mm(const float* lay, const float* a, float* zv, float* zg, float* sm, int lane, float* ws) {
  const float* st = stage(ws, lay + L_WV, H * H, lay + L_WG, H * H);
  const float* Wv = WSMEM ? st : lay + L_WV;
  const float* Wg = WSMEM ? st + H * H : lay + L_WG;
#pragma unroll
  for (int j = 0; j < CH; j++) { int c = lane + 32 * j; if (c < H) sm[c] = a[j]; }
  __syncwarp();
#pragma unroll
  for (int j = 0; j < CH; j++) { zv[j] = 0.0f; zg[j] = 0.0f; }
  for (int i = 0; i < H; i++) {
    float ai = sm[i];
#pragma unroll
    for (int j = 0; j < CH; j++) {
      int c = lane + 32 * j;
      if (c < H) { zv[j] += ai * rd(Wv + i * H + c); zg[j] += ai * rd(Wg + i * H + c); }
    }
  }
#pragma unroll
  for (int j = 0; j < CH; j++) { int c = lane + 32 * j; if (c < H) { zv[j] += ld(lay + L_BV + c); zg[j] += ld(lay + L_BG + c); } }
  __syncwarp();
}

// layer after its S4D output ys: residual GLU + layer norm. Saves what the backward needs.
struct LayerSave { float u[CH], pre[CH], zv[CH], zg[CH], nh[CH], inv; };

__device__ __forceinline__ void layer_fwd(const float* lay, float* x, const float* ys, float* sm, int lane, LayerSave& sv, float* ws) {
  float a[CH], zv[CH], zg[CH], v[CH];
#pragma unroll
  for (int j = 0; j < CH; j++) {
    int c = lane + 32 * j;
    sv.u[j] = x[j];
    sv.pre[j] = c < H ? ys[j] + ld(lay + L_D + c) * x[j] : 0.0f;
    a[j] = c < H ? gelu(sv.pre[j]) : 0.0f;
  }
  glu_mm(lay, a, zv, zg, sm, lane, ws);
  float m = 0.0f;
#pragma unroll
  for (int j = 0; j < CH; j++) {
    int c = lane + 32 * j;
    v[j] = c < H ? x[j] + zv[j] * sigm(zg[j]) : 0.0f;
    m += v[j];
    sv.zv[j] = zv[j]; sv.zg[j] = zg[j];
  }
  m = wsum(m) / H;
  float var = 0.0f;
#pragma unroll
  for (int j = 0; j < CH; j++) { int c = lane + 32 * j; float d = c < H ? v[j] - m : 0.0f; v[j] = d; var += d * d; }
  var = wsum(var) / H;
  sv.inv = 1.0f / sqrtf(var + 1e-5f);
#pragma unroll
  for (int j = 0; j < CH; j++) {
    int c = lane + 32 * j;
    sv.nh[j] = v[j] * sv.inv;
    x[j] = c < H ? sv.nh[j] * ld(lay + L_G + c) + ld(lay + L_BETA + c) : 0.0f;
  }
}

struct HeadSave { float ph[CHD]; float xl[CH]; };

// trunk output x -> decoder output o (without dec_b)
__device__ __forceinline__ float head_fwd(const float* kp, const float* x, float* sm, int lane, HeadSave& hs, float* ws) {
#pragma unroll
  for (int j = 0; j < CH; j++) hs.xl[j] = x[j];
  float o = 0.0f;
  if (HD > 0) {
    const float* st = stage(ws, kp + O_HW, H * HD);
    const float* hW = WSMEM ? st : kp + O_HW;
#pragma unroll
    for (int j = 0; j < CH; j++) { int c = lane + 32 * j; if (c < H) sm[c] = x[j]; }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < CHD; j++) {
      int c = lane + 32 * j;
      float acc = 0.0f;
      if (c < HD) {
        for (int i = 0; i < H; i++) acc += sm[i] * rd(hW + i * HD + c);
        acc += ld(kp + O_HB + c);
        o += gelu(acc) * ld(kp + O_DEC + c);
      }
      hs.ph[j] = acc;
    }
    __syncwarp();
  } else {
#pragma unroll
    for (int j = 0; j < CH; j++) { int c = lane + 32 * j; if (c < H) o += x[j] * ld(kp + O_DEC + c); }
  }
  return wsum(o);
}

// ======================================================================================= forward sweep
__global__ void __launch_bounds__(32 * WPB) fwd_kernel(const float* __restrict__ kp, const float* __restrict__ q, const float* __restrict__ st,
                                                       float* __restrict__ tape, float* __restrict__ ystape, float* __restrict__ sums, int B, int wtape) {
  __shared__ float smem[WPB][VMAX];
  __shared__ float ws[WSMEM ? WS_N : 1];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int b = blockIdx.x * WPB + wid;              // B is a multiple of WPB (all warps of a block meet at the staging barriers)
  float* sm = smem[wid];
  Patient P; P.load(q, b);
  const float u0 = ld(kp + O_SC), dec_b = ld(kp + O_SC + 1), w_low = ld(kp + O_SC + 2), rho_u = ld(kp + O_SC + 3);
  float x0 = 0, x1 = 0, x2 = 0, x3 = 0, ou = 0, up = 0, buf[DP];
#pragma unroll
  for (int j = 0; j < DP; j++) buf[j] = E0N;
  float hpr = 0.0f, hpi = 0.0f;                                     // mode `lane` (lane < PP)
  float hr[NL][CH][NS], hi[NL][CH][NS];
#pragma unroll
  for (int l = 0; l < NL; l++)
#pragma unroll
    for (int j = 0; j < CH; j++)
#pragma unroll
      for (int n = 0; n < NS; n++) { hr[l][j][n] = 0.0f; hi[l][j][n] = 0.0f; }
  float s_cost = 0.0f, s_se = 0.0f, s_v = 0.0f;
  const bool own_mode = lane < PP;
  const float lamr = own_mode ? ld(kp + O_LAMR + lane) : 0.0f, lami = own_mode ? ld(kp + O_LAMI + lane) : 0.0f;
  for (int k = 0; k < NSTEP; k++) {
    const size_t sb = (size_t)k * B + b;
    if (wtape) {
      float* tp = tape + sb * S;
      if (lane == 0) { tp[0] = x0; tp[1] = x1; tp[2] = x2; tp[3] = x3; tp[4] = ou; tp[5] = up; }
      if (own_mode) { tp[6 + lane] = hpr; tp[6 + PP + lane] = hpi; }
      if (lane < DP) {
        float bv = buf[0];
#pragma unroll
        for (int j = 1; j < DP; j++) if (lane == j) bv = buf[j];
        tp[6 + 2 * PP + lane] = bv;
      }
    }
    const float z0 = st[sb], z1 = st[(size_t)NSTEP * B + sb], tg = st[2 * (size_t)NSTEP * B + sb], sy = st[3 * (size_t)NSTEP * B + sb],
                r = st[4 * (size_t)NSTEP * B + sb], rho = st[5 * (size_t)NSTEP * B + sb], wk = st[6 * (size_t)NSTEP * B + sb];
    Pre pr;
    pre_stage(P, k, x3, ou, up, buf, z0, sy, tg, r, pr);
    float hqr = hpr * rho, hqi = hpi * rho;
    float x[CH];
#pragma unroll
    for (int j = 0; j < CH; j++) {
      int c = lane + 32 * j;
      x[j] = c < H ? pr.f0 * ld(kp + O_ENC + c) + pr.f1 * ld(kp + O_ENC + H + c) + pr.f2 * ld(kp + O_ENC + 2 * H + c) +
                     pr.f3 * ld(kp + O_ENC + 3 * H + c) + ld(kp + O_ENCB + c) : 0.0f;
    }
#pragma unroll
    for (int l = 0; l < NL; l++) {
      const float* lay = kp + O_LAY + l * L_STRIDE;
      float ys[CH];
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        float acc = 0.0f;
        if (c < H) {
#pragma unroll
          for (int n = 0; n < NS; n++) {
            const int o = c * NS + n;
            float Ar = ld(lay + L_AR + o), Ai = ld(lay + L_AI + o);
            float h2r = Ar * hr[l][j][n] - Ai * hi[l][j][n] + ld(lay + L_BR + o) * x[j];
            float h2i = Ar * hi[l][j][n] + Ai * hr[l][j][n] + ld(lay + L_BI + o) * x[j];
            hr[l][j][n] = h2r; hi[l][j][n] = h2i;
            acc += h2r * ld(lay + L_CR + o) - h2i * ld(lay + L_CI + o);
          }
          acc *= 2.0f;
          if (wtape) ystape[(((size_t)l * NSTEP + k) * B + b) * H + c] = acc;
        }
        ys[j] = acc;
      }
      LayerSave sv;
      layer_fwd(lay, x, ys, sm, lane, sv, ws);
    }
    HeadSave hs;
    float o = head_fwd(kp, x, sm, lane, hs, ws) + dec_b;
    float ul = wsum(own_mode ? ld(kp + O_COR + lane) * hqr - ld(kp + O_COI + lane) * hqi : 0.0f);
    float uu = fminf(fmaxf(r * (u0 + ul + 10.0f * o), 0.0f), UMAX);
    if (own_mode) {
      float un = uu / r, wn = pr.w / r;
      float nr = lamr * hqr - lami * hqi + ld(kp + O_BUR + lane) * un + ld(kp + O_BWR + lane) * wn;
      float ni = lamr * hqi + lami * hqr + ld(kp + O_BUI + lane) * un + ld(kp + O_BWI + lane) * wn;
      hpr = nr; hpi = ni;
    }
    float bb = pr.b, lo = relu(40.0f - bb);
    s_cost += wk * ((bb - tg) * (bb - tg) / 100.0f + w_low * lo * lo / 100.0f + rho_u * (uu / UMAX) * (uu / UMAX));
    s_se += (bb - tg) * (bb - tg);
    s_v += lo * lo;
    float n0 = P.A[0] * x0 + P.A[1] * x1 + P.A[2] * x2 + P.A[3] * x3 + P.Bd[0] * uu;
    float n1 = P.A[4] * x0 + P.A[5] * x1 + P.A[6] * x2 + P.A[7] * x3 + P.Bd[1] * uu;
    float n2 = P.A[8] * x0 + P.A[9] * x1 + P.A[10] * x2 + P.A[11] * x3 + P.Bd[2] * uu;
    float n3 = P.A[12] * x0 + P.A[13] * x1 + P.A[14] * x2 + P.A[15] * x3 + P.Bd[3] * uu;
    x0 = n0; x1 = n1; x2 = n2; x3 = n3;
    ou = P.a_ou * ou + P.osc * z1;
    up = uu;
  }
  if (lane == 0) { sums[b] = s_cost; sums[B + b] = s_se; sums[2 * B + b] = s_v; }
}

// ======================================================================================= backward sweep
__global__ void __launch_bounds__(32 * WPB) bwd_kernel(const float* __restrict__ kp, const float* __restrict__ q, const float* __restrict__ st,
                                                       const float* __restrict__ tape, const float* __restrict__ ystape, float* __restrict__ lamtape,
                                                       float* __restrict__ etape, int B) {
  __shared__ float smem[WPB][2 * VMAX];
  __shared__ float ws[WSMEM ? WS_N : 1];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int b = blockIdx.x * WPB + wid;
  float* sm = smem[wid];
  Patient P; P.load(q, b);
  const float u0 = ld(kp + O_SC), dec_b = ld(kp + O_SC + 1), w_low = ld(kp + O_SC + 2), rho_u = ld(kp + O_SC + 3);
  const float cbar = 1.0f / NSTEP;
  const bool own_mode = lane < PP;
  const float lamr = own_mode ? ld(kp + O_LAMR + lane) : 0.0f, lami = own_mode ? ld(kp + O_LAMI + lane) : 0.0f;
  // adjoint of the carry after the current step
  float lx0 = 0, lx1 = 0, lx2 = 0, lx3 = 0, lou = 0, lup = 0, lbuf[DP], lhr = 0, lhi = 0;
#pragma unroll
  for (int j = 0; j < DP; j++) lbuf[j] = 0.0f;
  float ahr[NL][CH][NS], ahi[NL][CH][NS];                           // S4D adjoints
#pragma unroll
  for (int l = 0; l < NL; l++)
#pragma unroll
    for (int j = 0; j < CH; j++)
#pragma unroll
      for (int n = 0; n < NS; n++) { ahr[l][j][n] = 0.0f; ahi[l][j][n] = 0.0f; }

  for (int k = NSTEP - 1; k >= 0; k--) {
    const size_t sb = (size_t)k * B + b;
    {   // tape the incoming adjoint
      float* lp = lamtape + sb * S;
      if (lane == 0) { lp[0] = lx0; lp[1] = lx1; lp[2] = lx2; lp[3] = lx3; lp[4] = lou; lp[5] = lup; }
      if (own_mode) { lp[6 + lane] = lhr; lp[6 + PP + lane] = lhi; }
      if (lane < DP) {
        float v = lbuf[0];
#pragma unroll
        for (int j = 1; j < DP; j++) if (lane == j) v = lbuf[j];
        lp[6 + 2 * PP + lane] = v;
      }
    }
    // ---- load the carry before step k and recompute the step (S4D outputs from the tape)
    const float* tp = tape + sb * S;
    float x0 = tp[0], x1 = tp[1], x2 = tp[2], x3 = tp[3], ou = tp[4], up = tp[5], buf[DP];
#pragma unroll
    for (int j = 0; j < DP; j++) buf[j] = tp[6 + 2 * PP + j];
    const float hpr = own_mode ? tp[6 + lane] : 0.0f, hpi = own_mode ? tp[6 + PP + lane] : 0.0f;
    const float z0 = st[sb], tg = st[2 * (size_t)NSTEP * B + sb], sy = st[3 * (size_t)NSTEP * B + sb],
                r = st[4 * (size_t)NSTEP * B + sb], rho = st[5 * (size_t)NSTEP * B + sb], wk = st[6 * (size_t)NSTEP * B + sb];
    Pre pr;
    pre_stage(P, k, x3, ou, up, buf, z0, sy, tg, r, pr);
    const float hqr = hpr * rho, hqi = hpi * rho;
    float x[CH];
#pragma unroll
    for (int j = 0; j < CH; j++) {
      int c = lane + 32 * j;
      x[j] = c < H ? pr.f0 * ld(kp + O_ENC + c) + pr.f1 * ld(kp + O_ENC + H + c) + pr.f2 * ld(kp + O_ENC + 2 * H + c) +
                     pr.f3 * ld(kp + O_ENC + 3 * H + c) + ld(kp + O_ENCB + c) : 0.0f;
    }
    LayerSave sv[NL];
    float e[NL][CH];
#pragma unroll
    for (int l = 0; l < NL; l++) {
      const float* lay = kp + O_LAY + l * L_STRIDE;
      float ys[CH];
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        float ev = 0.0f;
        if (c < H) {
          ys[j] = ystape[(((size_t)l * NSTEP + k) * B + b) * H + c];
#pragma unroll
          for (int n = 0; n < NS; n++) ev += ahr[l][j][n] * ld(lay + L_BR + c * NS + n) + ahi[l][j][n] * ld(lay + L_BI + c * NS + n);
          etape[(((size_t)l * NSTEP + k) * B + b) * H + c] = ev;
        } else {
          ys[j] = 0.0f;
        }
        e[l][j] = ev;
      }
      layer_fwd(lay, x, ys, sm, lane, sv[l], ws);
    }
    HeadSave hs;
    const float o = head_fwd(kp, x, sm, lane, hs, ws) + dec_b;
    const float ul = wsum(own_mode ? ld(kp + O_COR + lane) * hqr - ld(kp + O_COI + lane) * hqi : 0.0f);
    const float mean = r * (u0 + ul + 10.0f * o);
    const float uu = fminf(fmaxf(mean, 0.0f), UMAX);

    // ---- post stage backward
    float uub = P.Bd[0] * lx0 + P.Bd[1] * lx1 + P.Bd[2] * lx2 + P.Bd[3] * lx3 + lup + cbar * wk * rho_u * 2.0f * uu / (UMAX * UMAX);
    uub += wsum(own_mode ? lhr * ld(kp + O_BUR + lane) + lhi * ld(kp + O_BUI + lane) : 0.0f) / r;
    float wb = wsum(own_mode ? lhr * ld(kp + O_BWR + lane) + lhi * ld(kp + O_BWI + lane) : 0.0f) / r;
    float hqrb = lamr * lhr + lami * lhi, hqib = lamr * lhi - lami * lhr;
    const float bb = pr.b;
    float bbar = cbar * wk * (2.0f * (bb - tg) / 100.0f - w_low * 2.0f * relu(40.0f - bb) / 100.0f);
    const float meanb = (mean > 0.0f && mean < UMAX) ? uub : 0.0f;
    const float ulb = meanb * r, ob = meanb * r * 10.0f;
    if (own_mode) { hqrb += ulb * ld(kp + O_COR + lane); hqib -= ulb * ld(kp + O_COI + lane); }
    float xb[CH];
    if (HD > 0) {
      const float* st = stage(ws, kp + O_HWT, HD * H);
      const float* hWT = WSMEM ? st : kp + O_HWT;
#pragma unroll
      for (int j = 0; j < CHD; j++) { int c = lane + 32 * j; if (c < HD) sm[c] = ob * ld(kp + O_DEC + c) * gelu_d(hs.ph[j]); }
      __syncwarp();
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        float acc = 0.0f;
        if (c < H) for (int i = 0; i < HD; i++) acc += sm[i] * rd(hWT + i * H + c);
        xb[j] = acc;
      }
      __syncwarp();
    } else {
#pragma unroll
      for (int j = 0; j < CH; j++) { int c = lane + 32 * j; xb[j] = c < H ? ob * ld(kp + O_DEC + c) : 0.0f; }
    }
    // ---- layers backward
#pragma unroll
    for (int l = NL - 1; l >= 0; l--) {
      const float* lay = kp + O_LAY + l * L_STRIDE;
      const LayerSave& s = sv[l];
      const float* stT = stage(ws, lay + L_WVT, H * H, lay + L_WGT, H * H);
      const float* WvT = WSMEM ? stT : lay + L_WVT;
      const float* WgT = WSMEM ? stT + H * H : lay + L_WGT;
      float nb[CH], m1 = 0.0f, m2 = 0.0f;
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        nb[j] = c < H ? xb[j] * ld(lay + L_G + c) : 0.0f;
        m1 += nb[j]; m2 += nb[j] * s.nh[j];
      }
      m1 = wsum(m1) / H; m2 = wsum(m2) / H;
      float zvb[CH], zgb[CH], ub[CH];
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        float vb = c < H ? s.inv * (nb[j] - m1 - s.nh[j] * m2) : 0.0f;
        float sg = sigm(s.zg[j]);
        ub[j] = vb;
        zvb[j] = vb * sg;
        zgb[j] = vb * s.zv[j] * sg * (1.0f - sg);
        if (c < H) { sm[c] = zvb[j]; sm[VMAX + c] = zgb[j]; }
      }
      __syncwarp();
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        if (c < H) {
          float ab = 0.0f;
          for (int i = 0; i < H; i++) ab += sm[i] * rd(WvT + i * H + c) + sm[VMAX + i] * rd(WgT + i * H + c);
          float preb = ab * gelu_d(s.pre[j]);
          float cy = preb;
          ub[j] += preb * ld(lay + L_D + c) + cy * ld(lay + L_K0 + c) + e[l][j];
#pragma unroll
          for (int n = 0; n < NS; n++) {
            const int oo = c * NS + n;
            float gr = ahr[l][j][n] + 2.0f * cy * ld(lay + L_CR + oo);
            float gi = ahi[l][j][n] - 2.0f * cy * ld(lay + L_CI + oo);
            float Ar = ld(lay + L_AR + oo), Ai = ld(lay + L_AI + oo);
            ahr[l][j][n] = gr * Ar + gi * Ai;
            ahi[l][j][n] = gi * Ar - gr * Ai;
          }
        }
        xb[j] = ub[j];
      }
      __syncwarp();
    }
    // ---- pre stage backward
    float fb0 = 0.0f, fb1 = 0.0f, fb3 = 0.0f;
#pragma unroll
    for (int j = 0; j < CH; j++) {
      int c = lane + 32 * j;
      if (c < H) { fb0 += xb[j] * ld(kp + O_ENC + c); fb1 += xb[j] * ld(kp + O_ENC + H + c); fb3 += xb[j] * ld(kp + O_ENC + 3 * H + c); }
    }
    fb0 = wsum(fb0); fb1 = wsum(fb1); fb3 = wsum(fb3);
    float yb = fb0 / 50.0f;
    const float upb = fb1 / (r * UMAX);
    wb += fb3 / (r * C50N);
    const float w0b = pr.w0 < 3.0f * C50N ? wb : 0.0f;
    const float ratio = pr.s / (1.0f - pr.s);
    const float sbar = w0b * C50N * (1.0f / GAMN) * powf(ratio, 1.0f / GAMN - 1.0f) / ((1.0f - pr.s) * (1.0f - pr.s));
    const float s0b = (pr.s0 > 1e-3f && pr.s0 < 1.0f - 1e-3f) ? sbar : 0.0f;
    yb += -s0b / EMAXN;
    float bufb[DP];
#pragma unroll
    for (int j = 0; j < DP; j++) bufb[j] = lbuf[j] + yb * P.dmask[j];
    if (k == 0) {
#pragma unroll
      for (int j = 0; j < DP; j++) { bbar += bufb[j]; lbuf[j] = 0.0f; }
    } else {
      bbar += bufb[0];
#pragma unroll
      for (int j = 0; j < DP - 1; j++) lbuf[j] = bufb[j + 1];
      lbuf[DP - 1] = 0.0f;
    }
    const float den = pr.cg + pr.eg;
    const float cgb = -bbar * P.emax * pr.eg / (den * den);
    const float egb = bbar * P.emax * pr.cg / (den * den);
    const float x3b = (x3 > 0.0f) ? cgb * P.gam * powf(pr.ce, P.gam - 1.0f) : 0.0f;
    // ---- adjoint of the carry before step k
    const float nx0 = P.A[0] * lx0 + P.A[4] * lx1 + P.A[8] * lx2 + P.A[12] * lx3;
    const float nx1 = P.A[1] * lx0 + P.A[5] * lx1 + P.A[9] * lx2 + P.A[13] * lx3;
    const float nx2 = P.A[2] * lx0 + P.A[6] * lx1 + P.A[10] * lx2 + P.A[14] * lx3;
    const float nx3 = P.A[3] * lx0 + P.A[7] * lx1 + P.A[11] * lx2 + P.A[15] * lx3 + x3b;
    lx0 = nx0; lx1 = nx1; lx2 = nx2; lx3 = nx3;
    lou = P.a_ou * lou + egb * P.gam * pr.eg;
    lup = upb;
    lhr = hqrb * rho; lhi = hqib * rho;
  }
}

// ======================================================================================= FFI
static ffi::Error Fwd(cudaStream_t stream, ffi::Buffer<ffi::F32> kp, ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> st,
                      ffi::ResultBuffer<ffi::F32> tape, ffi::ResultBuffer<ffi::F32> ys, ffi::ResultBuffer<ffi::F32> sums, int32_t wtape) {
  const int B = (int)q.dimensions()[0];
  fwd_kernel<<<(B + WPB - 1) / WPB, 32 * WPB, 0, stream>>>(kp.typed_data(), q.typed_data(), st.typed_data(), tape->typed_data(), ys->typed_data(),
                                                         sums->typed_data(), B, wtape);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
  return ffi::Error::Success();
}

static ffi::Error Bwd(cudaStream_t stream, ffi::Buffer<ffi::F32> kp, ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> st, ffi::Buffer<ffi::F32> tape,
                      ffi::Buffer<ffi::F32> ys, ffi::ResultBuffer<ffi::F32> lam, ffi::ResultBuffer<ffi::F32> e) {
  const int B = (int)q.dimensions()[0];
  bwd_kernel<<<(B + WPB - 1) / WPB, 32 * WPB, 0, stream>>>(kp.typed_data(), q.typed_data(), st.typed_data(), tape.typed_data(), ys.typed_data(),
                                                         lam->typed_data(), e->typed_data(), B);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PkpdFwd, Fwd,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
                                  .Attr<int32_t>("tape"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(PkpdBwd, Bwd,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());
