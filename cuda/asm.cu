// CUDA gradient assembly of kstep.py (the per-patient parameter gradients from the two tapes), called from JAX as an XLA FFI custom call.
// Every (patient, step) is independent given the tapes: the warp of a (patient, segment of steps) replays each step of its segment from the forward
// tape (carry before the step, S4D outputs), takes the step's VJP with the carry adjoint and the S4D-recurrence cotangents of the backward tape, and
//   - accumulates the cotangents of every small parameter (vectors) over its steps, in registers;
//   - writes per step the operands of the large-matrix gradients (the GLU input a and the cotangents of zv, zg per layer; the head input and the
//     cotangent of its pre-activation) and of the S4D-kernel gradients (layer input u and S4D output cotangent cy), which kstep.py contracts with
//     batched GEMMs and an FFT cross-correlation.
// Masked layout (padded population members) and plain layout (one unpadded network) share the code: the plain layout passes all-ones masks.
// Sizes are compile-time constants (-DH=.. -DNS=.. -DNL=.. -DHD=.. -DPP=.. -DWPB=.. -DNSEG=.. -DSTAGE=0|1); kcuda.py builds one library per shape.
// Layouts (kcuda.py packs / unpacks them):
//   kp   (M, KTOT): sc[16] = u0 dec_b w_low rho_u nH head gate[NL] | lamr lami Bur Bui Bwr Bwi Cor Coi [PP each] | enc0..enc3 enc_b mH decl [H each] |
//        dec [HDV] | (HD > 0) hW [H*HD] hWT [HD*H] hb [HD] | per layer: K0 D [H] Wv WvT Wg WgT [H*H] bv bg g beta [H]
//   q    (M, B, 48), st (M, 6, N, B), tape / lam (M, N, B, S), ys / e (M, NL, N, B, H)    (M: population members; broadcast inputs are copies)
//   gs   (M, NSEG, B, NSM): small-parameter cotangents per segment: u0 dec_b | lamr lami Bur Bui Bwr Bwi Cor Coi [PP] | enc0..3 enc_b decl [H] |
//        dec [HDV] | hb [HD] | per layer: D bv bg g beta [H]
//   ops  (M, B, N, NOP): per layer: a zvb zgb u cy [H each] | head input xl [H] | head pre-activation cotangent dph [HD]
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#ifndef H
#error "compile with -DH=.. -DNS=.. -DNL=.. -DHD=.. -DPP=.. -DWPB=.. -DNSEG=.."
#endif
#ifndef STAGE
#define STAGE 0
#endif
#ifndef NSTEP
#define NSTEP 360
#endif
constexpr int DP = 16, NQ = 48, NSC = 16;
constexpr float DT = 5.0f / 60.0f, T_END = 30.0f, UMAX = 100.0f, E0N = 100.0f, EMAXN = 100.0f, GAMN = 2.0f, C50N = 4.5f;
constexpr int CH = (H + 31) / 32;
constexpr int HDV = HD > 0 ? HD : H;
constexpr int CHD = (HDV + 31) / 32;
constexpr int S = 6 + 2 * PP + DP;
constexpr int VMAX = (H > HDV ? H : HDV);
constexpr int SEGLEN = NSTEP / NSEG;
static_assert(NSTEP % NSEG == 0, "NSEG must divide the episode");
constexpr int WS_N = (2 * H * H > H * HD ? 2 * H * H : H * HD);
constexpr bool WSMEM = STAGE && WS_N * 4 <= 40 * 1024;
// parameter offsets
constexpr int O_SC = 0, O_LAMR = NSC, O_LAMI = O_LAMR + PP, O_BUR = O_LAMI + PP, O_BUI = O_BUR + PP, O_BWR = O_BUI + PP, O_BWI = O_BWR + PP,
              O_COR = O_BWI + PP, O_COI = O_COR + PP, O_ENC = O_COI + PP, O_ENCB = O_ENC + 4 * H, O_MH = O_ENCB + H, O_DECL = O_MH + H,
              O_DEC = O_DECL + H, O_HW = O_DEC + HDV, O_HWT = O_HW + H * HD, O_HB = O_HWT + HD * H, O_LAY = O_HB + HD;
constexpr int L_K0 = 0, L_D = H, L_WV = 2 * H, L_WVT = L_WV + H * H, L_WG = L_WVT + H * H, L_WGT = L_WG + H * H, L_BV = L_WGT + H * H,
              L_BG = L_BV + H, L_G = L_BG + H, L_BETA = L_G + H, L_STRIDE = L_BETA + H;
constexpr int KTOT = O_LAY + NL * L_STRIDE;
// small-gradient offsets
constexpr int G_U0 = 0, G_DECB = 1, G_LAMR = 2, G_ENC = G_LAMR + 8 * PP, G_ENCB = G_ENC + 4 * H, G_DECL = G_ENCB + H, G_DEC = G_DECL + H,
              G_HB = G_DEC + HDV, G_LAY = G_HB + HD, NSM = G_LAY + NL * 5 * H;
// operand offsets (per step)
constexpr int P_LAY = 5 * H, P_XL = NL * P_LAY, P_DPH = P_XL + H, NOP = P_DPH + HD;

__device__ __forceinline__ float ld(const float* p) { return __ldg(p); }

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

struct Patient {
  float mu, a_ou, osc, stim, gam, e0, emax, dmask[DP], A[16], Bd[4], stim_t, stim2, stim_t2, stim3, stim_t3;
  __device__ void load(const float* r) {
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

// monitor, delay line, inverse Hill, features of step k from the carry before it (only what the gradients need)
__device__ __forceinline__ void pre_stage(const Patient& P, int k, float x3, float ou, float up, const float* buf_in, float z0, float sy, float tg,
                                          float r, float& b, float& w, float& f0, float& f1, float& f2, float& f3) {
  float t = k * DT;
  float ce = fmaxf(x3, 0.0f);
  float l50 = P.mu + ou + P.stim * (t > P.stim_t ? 1.0f : 0.0f) + P.stim2 * (t > P.stim_t2 ? 1.0f : 0.0f) + P.stim3 * (t > P.stim_t3 ? 1.0f : 0.0f);
  float cg = powf(ce, P.gam), eg = powf(expf(l50), P.gam);
  b = P.e0 - P.emax * cg / (cg + eg);
  float bd = 0.0f;
  if (k == 0) {
#pragma unroll
    for (int j = 0; j < DP; j++) bd += b * P.dmask[j];
  } else {
    bd = b * P.dmask[0];
#pragma unroll
    for (int j = 1; j < DP; j++) bd += buf_in[j - 1] * P.dmask[j];
  }
  float y = bd + sy * z0;
  float s = fminf(fmaxf((E0N - y) / EMAXN, 1e-3f), 1.0f - 1e-3f);
  w = fminf(C50N * powf(s / (1.0f - s), 1.0f / GAMN), 3.0f * C50N);
  f0 = (y - tg) / 50.0f;
  f1 = up / (r * UMAX);
  f2 = 0.0f;
  f3 = w / (r * C50N) - 1.0f;
}

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

struct LayerSave { float u[CH], pre[CH], zv[CH], zg[CH], nh[CH], inv; };

// residual GLU + masked layer norm (gate off: the layer is skipped)
__device__ __forceinline__ void layer_fwd(const float* lay, float* x, const float* ys, const float* mH, float nH, bool gate, float* sm, int lane,
                                          LayerSave& sv, float* ws) {
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
    m += v[j] * mH[j];
    sv.zv[j] = zv[j]; sv.zg[j] = zg[j];
  }
  m = wsum(m) / nH;
  float var = 0.0f;
#pragma unroll
  for (int j = 0; j < CH; j++) { float d = (v[j] - m) * mH[j]; v[j] = d; var += d * d; }
  var = wsum(var) / nH;
  sv.inv = 1.0f / sqrtf(var + 1e-5f);
#pragma unroll
  for (int j = 0; j < CH; j++) {
    int c = lane + 32 * j;
    sv.nh[j] = v[j] * sv.inv;
    if (gate) x[j] = c < H ? sv.nh[j] * ld(lay + L_G + c) + ld(lay + L_BETA + c) : 0.0f;
  }
}

// ======================================================================================= assembly kernel
__global__ void __launch_bounds__(32 * WPB) asm_kernel(const float* __restrict__ kp_all, const float* __restrict__ q_all, const float* __restrict__ st_all,
                                                       const float* __restrict__ tape_all, const float* __restrict__ ys_all,
                                                       const float* __restrict__ lam_all, const float* __restrict__ e_all,
                                                       float* __restrict__ gs_all, float* __restrict__ ops_all, int B) {
  __shared__ float smem[WPB][2 * VMAX];
  __shared__ float ws[WSMEM ? WS_N : 1];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int m = blockIdx.y;
  const int item = blockIdx.x * WPB + wid;                   // (segment, patient): B * NSEG is a multiple of WPB
  const int seg = item / B, b = item % B;
  const float* kp = kp_all + (size_t)m * KTOT;
  const float* q = q_all + (size_t)m * B * NQ;
  const float* st = st_all + (size_t)m * 7 * NSTEP * B;
  const float* tape = tape_all + (size_t)m * NSTEP * B * S;
  const float* lamt = lam_all + (size_t)m * NSTEP * B * S;
  const float* yst = ys_all + (size_t)m * NL * NSTEP * B * H;
  const float* et = e_all + (size_t)m * NL * NSTEP * B * H;
  float* gs = gs_all + (((size_t)m * NSEG + seg) * B + b) * NSM;
  float* sm = smem[wid];
  Patient P; P.load(q + (size_t)b * NQ);
  const float u0 = ld(kp + O_SC), dec_b = ld(kp + O_SC + 1), w_low = ld(kp + O_SC + 2), rho_u = ld(kp + O_SC + 3), nH = ld(kp + O_SC + 4);
  const bool head = HD > 0 && ld(kp + O_SC + 5) > 0.0f;
  const float cbar = 1.0f / NSTEP;                                   // (times the step's weight wk)
  const bool own_mode = lane < PP;
  const float lamr = own_mode ? ld(kp + O_LAMR + lane) : 0.0f, lami = own_mode ? ld(kp + O_LAMI + lane) : 0.0f;
  float mH[CH];
#pragma unroll
  for (int j = 0; j < CH; j++) { int c = lane + 32 * j; mH[j] = c < H ? ld(kp + O_MH + c) : 0.0f; }
  // accumulators
  float g_u0 = 0.0f, g_decb = 0.0f, g_ph[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float g_enc[5][CH], g_decl[CH], g_dec[CHD], g_hb[CHD], g_lay[NL][5][CH];
#pragma unroll
  for (int j = 0; j < CH; j++) {
    g_decl[j] = 0.0f;
#pragma unroll
    for (int i = 0; i < 5; i++) g_enc[i][j] = 0.0f;
#pragma unroll
    for (int l = 0; l < NL; l++)
#pragma unroll
      for (int i = 0; i < 5; i++) g_lay[l][i][j] = 0.0f;
  }
#pragma unroll
  for (int j = 0; j < CHD; j++) { g_dec[j] = 0.0f; g_hb[j] = 0.0f; }

  for (int k = seg * SEGLEN; k < (seg + 1) * SEGLEN; k++) {
    const size_t sb = (size_t)k * B + b;
    float* op = ops_all + (((size_t)m * B + b) * NSTEP + k) * NOP;
    const float* tp = tape + sb * S;
    const float* lp = lamt + sb * S;
    const float x3 = tp[3], ou = tp[4], up = tp[5];
    float buf[DP];
#pragma unroll
    for (int j = 0; j < DP; j++) buf[j] = tp[6 + 2 * PP + j];
    const float hpr = own_mode ? tp[6 + lane] : 0.0f, hpi = own_mode ? tp[6 + PP + lane] : 0.0f;
    const float z0 = st[sb], tg = st[2 * (size_t)NSTEP * B + sb], sy = st[3 * (size_t)NSTEP * B + sb],
                r = st[4 * (size_t)NSTEP * B + sb], rho = st[5 * (size_t)NSTEP * B + sb], wk = st[6 * (size_t)NSTEP * B + sb];
    float bb, w, f0, f1, f2, f3;
    pre_stage(P, k, x3, ou, up, buf, z0, sy, tg, r, bb, w, f0, f1, f2, f3);
    const float hqr = hpr * rho, hqi = hpi * rho;
    float x[CH];
#pragma unroll
    for (int j = 0; j < CH; j++) {
      int c = lane + 32 * j;
      x[j] = c < H ? f0 * ld(kp + O_ENC + c) + f1 * ld(kp + O_ENC + H + c) + f2 * ld(kp + O_ENC + 2 * H + c) +
                     f3 * ld(kp + O_ENC + 3 * H + c) + ld(kp + O_ENCB + c) : 0.0f;
    }
    LayerSave sv[NL];
#pragma unroll
    for (int l = 0; l < NL; l++) {
      const float* lay = kp + O_LAY + l * L_STRIDE;
      const bool gate = ld(kp + O_SC + 6 + l) > 0.0f;
      float ys[CH];
#pragma unroll
      for (int j = 0; j < CH; j++) { int c = lane + 32 * j; ys[j] = c < H ? yst[(((size_t)l * NSTEP + k) * B + b) * H + c] : 0.0f; }
      layer_fwd(lay, x, ys, mH, nH, gate, sm, lane, sv[l], ws);
    }
    // ---- head forward
    float ph[CHD], o = 0.0f;
#pragma unroll
    for (int j = 0; j < CHD; j++) ph[j] = 0.0f;
    if (head) {
      const float* stg = stage(ws, kp + O_HW, H * HD);
      const float* hW = WSMEM ? stg : kp + O_HW;
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
        ph[j] = acc;
      }
      __syncwarp();
    } else if (HD > 0) {
#pragma unroll
      for (int j = 0; j < CH; j++) { int c = lane + 32 * j; if (c < H) o += x[j] * ld(kp + O_DECL + c); }
    } else {
#pragma unroll
      for (int j = 0; j < CH; j++) { int c = lane + 32 * j; if (c < H) o += x[j] * ld(kp + O_DEC + c); }
    }
    o = wsum(o) + dec_b;
    const float ul = wsum(own_mode ? ld(kp + O_COR + lane) * hqr - ld(kp + O_COI + lane) * hqi : 0.0f);
    const float mean = r * (u0 + ul + 10.0f * o);
    const float uu = fminf(fmaxf(mean, 0.0f), UMAX);

    // ---- post stage backward (carry adjoint after the step from the backward tape)
    const float lx0 = lp[0], lx1 = lp[1], lx2 = lp[2], lx3 = lp[3], lup = lp[5];
    const float lhr = own_mode ? lp[6 + lane] : 0.0f, lhi = own_mode ? lp[6 + PP + lane] : 0.0f;
    float uub = P.Bd[0] * lx0 + P.Bd[1] * lx1 + P.Bd[2] * lx2 + P.Bd[3] * lx3 + lup + cbar * wk * rho_u * 2.0f * uu / (UMAX * UMAX);
    uub += wsum(own_mode ? lhr * ld(kp + O_BUR + lane) + lhi * ld(kp + O_BUI + lane) : 0.0f) / r;
    const float meanb = (mean > 0.0f && mean < UMAX) ? uub : 0.0f;
    const float ulb = meanb * r, ob = meanb * r * 10.0f;
    g_u0 += meanb * r;
    g_decb += ob;
    if (own_mode) {
      const float un = uu / r, wn = w / r;
      g_ph[0] += lhr * hqr + lhi * hqi;          // lamr
      g_ph[1] += -lhr * hqi + lhi * hqr;         // lami
      g_ph[2] += lhr * un;                       // Bur
      g_ph[3] += lhi * un;                       // Bui
      g_ph[4] += lhr * wn;                       // Bwr
      g_ph[5] += lhi * wn;                       // Bwi
      g_ph[6] += ulb * hqr;                      // Cor
      g_ph[7] += -ulb * hqi;                     // Coi
    }
    float xb[CH];
    if (head) {
      const float* stg = stage(ws, kp + O_HWT, HD * H);
      const float* hWT = WSMEM ? stg : kp + O_HWT;
#pragma unroll
      for (int j = 0; j < CHD; j++) {
        int c = lane + 32 * j;
        if (c < HD) {
          float d = ob * ld(kp + O_DEC + c) * gelu_d(ph[j]);
          g_dec[j] += ob * gelu(ph[j]);
          g_hb[j] += d;
          sm[c] = d;
          op[P_DPH + c] = d;
        }
      }
      __syncwarp();
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        float acc = 0.0f;
        if (c < H) {
          for (int i = 0; i < HD; i++) acc += sm[i] * rd(hWT + i * H + c);
          op[P_XL + c] = x[j];
        }
        xb[j] = acc;
      }
      __syncwarp();
    } else {
      const int odec = HD > 0 ? O_DECL : O_DEC;
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        xb[j] = c < H ? ob * ld(kp + odec + c) : 0.0f;
        if (c < H) {
          if (HD > 0) g_decl[j] += ob * x[j]; else g_dec[j] += ob * x[j];
          op[P_XL + c] = x[j];
        }
      }
      if (HD > 0)
#pragma unroll
        for (int j = 0; j < CHD; j++) { int c = lane + 32 * j; if (c < HD) op[P_DPH + c] = 0.0f; }
    }
    // ---- layers backward
#pragma unroll
    for (int l = NL - 1; l >= 0; l--) {
      const float* lay = kp + O_LAY + l * L_STRIDE;
      const LayerSave& s = sv[l];
      float* ol = op + l * P_LAY;
      const bool gate = ld(kp + O_SC + 6 + l) > 0.0f;
      if (!gate) {                       // skipped layer: the cotangent passes through; nothing of the layer is used
#pragma unroll
        for (int j = 0; j < CH; j++) {
          int c = lane + 32 * j;
          if (c < H) {
            xb[j] += et[(((size_t)l * NSTEP + k) * B + b) * H + c];
            ol[c] = 0.0f; ol[H + c] = 0.0f; ol[2 * H + c] = 0.0f; ol[3 * H + c] = s.u[j]; ol[4 * H + c] = 0.0f;
          }
        }
        continue;
      }
      const float* stT = stage(ws, lay + L_WVT, H * H, lay + L_WGT, H * H);
      const float* WvT = WSMEM ? stT : lay + L_WVT;
      const float* WgT = WSMEM ? stT + H * H : lay + L_WGT;
      float nb[CH], m1 = 0.0f, m2 = 0.0f;
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        if (c < H) { g_lay[l][3][j] += xb[j] * s.nh[j]; g_lay[l][4][j] += xb[j]; }
        nb[j] = c < H ? xb[j] * ld(lay + L_G + c) * mH[j] : 0.0f;
        m1 += nb[j]; m2 += nb[j] * s.nh[j];
      }
      m1 = wsum(m1) / nH; m2 = wsum(m2) / nH;
      float zvb[CH], zgb[CH], ub[CH];
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        float vb = c < H ? mH[j] * s.inv * (nb[j] - m1 - s.nh[j] * m2) : 0.0f;
        float sg = sigm(s.zg[j]);
        ub[j] = vb;
        zvb[j] = vb * sg;
        zgb[j] = vb * s.zv[j] * sg * (1.0f - sg);
        if (c < H) {
          sm[c] = zvb[j]; sm[VMAX + c] = zgb[j];
          g_lay[l][1][j] += zvb[j]; g_lay[l][2][j] += zgb[j];
          ol[c] = gelu(s.pre[j]); ol[H + c] = zvb[j]; ol[2 * H + c] = zgb[j]; ol[3 * H + c] = s.u[j];
        }
      }
      __syncwarp();
#pragma unroll
      for (int j = 0; j < CH; j++) {
        int c = lane + 32 * j;
        if (c < H) {
          float ab = 0.0f;
          for (int i = 0; i < H; i++) ab += sm[i] * rd(WvT + i * H + c) + sm[VMAX + i] * rd(WgT + i * H + c);
          float preb = ab * gelu_d(s.pre[j]);
          g_lay[l][0][j] += preb * s.u[j];
          ol[4 * H + c] = preb;
          ub[j] += preb * ld(lay + L_D + c) + preb * ld(lay + L_K0 + c) + et[(((size_t)l * NSTEP + k) * B + b) * H + c];
        }
        xb[j] = ub[j];
      }
      __syncwarp();
    }
    // ---- encoder
#pragma unroll
    for (int j = 0; j < CH; j++) {
      int c = lane + 32 * j;
      if (c < H) { g_enc[0][j] += xb[j] * f0; g_enc[1][j] += xb[j] * f1; g_enc[2][j] += xb[j] * f2; g_enc[3][j] += xb[j] * f3; g_enc[4][j] += xb[j]; }
    }
  }
  // ---- write the segment's small-parameter cotangents
  if (lane == 0) { gs[G_U0] = g_u0; gs[G_DECB] = g_decb; }
  if (own_mode)
#pragma unroll
    for (int i = 0; i < 8; i++) gs[G_LAMR + i * PP + lane] = g_ph[i];
#pragma unroll
  for (int j = 0; j < CH; j++) {
    int c = lane + 32 * j;
    if (c < H) {
#pragma unroll
      for (int i = 0; i < 5; i++) gs[G_ENC + i * H + c] = g_enc[i][j];
      gs[G_DECL + c] = g_decl[j];
#pragma unroll
      for (int l = 0; l < NL; l++)
#pragma unroll
        for (int i = 0; i < 5; i++) gs[G_LAY + (l * 5 + i) * H + c] = g_lay[l][i][j];
    }
  }
#pragma unroll
  for (int j = 0; j < CHD; j++) {
    int c = lane + 32 * j;
    if (c < HDV) gs[G_DEC + c] = g_dec[j];
    if (c < HD) gs[G_HB + c] = g_hb[j];
  }
}

// ======================================================================================= FFI
static ffi::Error Asm(cudaStream_t stream, ffi::Buffer<ffi::F32> kp, ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> st, ffi::Buffer<ffi::F32> tape,
                      ffi::Buffer<ffi::F32> ys, ffi::Buffer<ffi::F32> lam, ffi::Buffer<ffi::F32> e, ffi::ResultBuffer<ffi::F32> gs,
                      ffi::ResultBuffer<ffi::F32> ops) {
  const auto qd = q.dimensions();
  const int B = (int)qd[qd.size() - 2];
  const int M = (int)(kp.element_count() / KTOT);
  if ((B * NSEG) % WPB != 0) return ffi::Error(ffi::ErrorCode::kInvalidArgument, "B * NSEG must be a multiple of WPB");
  dim3 grid((B * NSEG) / WPB, M);
  asm_kernel<<<grid, 32 * WPB, 0, stream>>>(kp.typed_data(), q.typed_data(), st.typed_data(), tape.typed_data(), ys.typed_data(), lam.typed_data(),
                                            e.typed_data(), gs->typed_data(), ops->typed_data(), B);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PkpdAsm, Asm,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());
