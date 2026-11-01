// casm.cu -- the tape assembly (cbwd.assemble_tapes) as one CUDA kernel: per patient, one pass over the gradient tapes that cuda/cbwd.cu writes.
// It gives the per-patient gradients of the GLU and head matrices (sums over the steps of activation x cotangent outer products), the GLU
// biases and D, and the two moments of every S4D mode (cuda/s4grad.cu's P = sum_k cy_k s_k and P' = sum_k cy_k d_k with s_k = dA s_(k-1) + u_k,
// d_k = dA d_(k-1) + s_(k-1)). Thread block = (mode slice, patient, member): the steps are streamed through shared memory in chunks of CK, so
// every tape is read from memory once (a patient's slices run together and share t_x in L2); slice 0 also accumulates the matrices and sums,
// every slice a share of the modes (RM per thread).
// Compile-time configuration: cfg.h (cbwd.py).
#include <cuda_runtime.h>
#include <stdint.h>
#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

#define WX (2 * LP * HP)                         // row of t_x: per layer (u, cy)
#define NM (LP * HP * NSP)                       // S4D modes
#define HQ (HP / 4)
#define RW ((HP * HQ + NT - 1) / NT)              // (row, 4 columns) items per thread of each GLU matrix pair
#define RH ((HP * (HDP / 4) + NT - 1) / NT)       // ... of the head matrix
#define RB ((3 * LP * HP + NT - 1) / NT)          // bias / D sums per thread
#define RHB ((HDP + NT - 1) / NT)

struct Args { const float* in[NIN]; float* out[NOUT]; };
#define INP(n) (A.in[I_##n])
#define OUT(n) (A.out[O_##n])

__device__ __forceinline__ void cp(float* dst, const float* __restrict__ src, int n) {         // n a multiple of 4, 16-byte aligned
  for (int i = threadIdx.x; i < n / 4; i += NT) reinterpret_cast<float4*>(dst)[i] = __ldg(reinterpret_cast<const float4*>(src) + i);
}

__global__ void __launch_bounds__(NT) casm_kernel(Args A, int B) {
  extern __shared__ float4 sm4[];
  float* SX = reinterpret_cast<float*>(sm4);    // [CK][WX]       (u, cy) per layer
  float* SA = SX + CK * WX;                     // [LP][3][CK][HP] a, dzv, dzg
  float* SE = SA + LP * 3 * CK * HP;            // [CK][HP]       trunk output xe
  float* SD = SE + CK * HP;                     // [CK][HDP]      head pre-activation cotangent
  const int t = threadIdx.x, slice = blockIdx.x, b = blockIdx.y, g = blockIdx.z;       // a patient's slices adjacent: t_x shared in L2
  const size_t gb = (size_t)g * B + b;
  const bool mats = slice == 0;

  float aw[LP][RW][2][4], ah[RH][4], ab[RB], ahb[RHB];
#pragma unroll
  for (int l = 0; l < LP; l++)
#pragma unroll
    for (int r = 0; r < RW; r++)
#pragma unroll
      for (int c = 0; c < 4; c++) { aw[l][r][0][c] = 0.f; aw[l][r][1][c] = 0.f; }
#pragma unroll
  for (int r = 0; r < RH; r++)
#pragma unroll
    for (int c = 0; c < 4; c++) ah[r][c] = 0.f;
#pragma unroll
  for (int r = 0; r < RB; r++) ab[r] = 0.f;
#pragma unroll
  for (int r = 0; r < RHB; r++) ahb[r] = 0.f;
  // this thread's modes: m = (slice * RM + r) * NT + t
  float Ar[RM], Ai[RM], sr[RM], si[RM], dr[RM], di[RM], pr[RM], pi[RM], qr[RM], qi[RM];
#pragma unroll
  for (int r = 0; r < RM; r++) {
    const int m = (slice * RM + r) * NT + t;
    const bool ok = m < NM;
    Ar[r] = ok ? __ldg(INP(dA) + ((size_t)g * NM + m) * 2) : 0.f;
    Ai[r] = ok ? __ldg(INP(dA) + ((size_t)g * NM + m) * 2 + 1) : 0.f;
    sr[r] = si[r] = dr[r] = di[r] = pr[r] = pi[r] = qr[r] = qi[r] = 0.f;
  }

  for (int k0 = 0; k0 < NSTEP; k0 += CK) {
    __syncthreads();
    cp(SX, INP(t_x) + (gb * NSTEP + k0) * WX, CK * WX);
    if (mats) {
#pragma unroll
      for (int l = 0; l < LP; l++) {
        const size_t o = ((((size_t)g * LP + l) * B + b) * NSTEP + k0) * HP;
        cp(SA + (l * 3 + 0) * CK * HP, INP(t_a) + o, CK * HP);
        cp(SA + (l * 3 + 1) * CK * HP, INP(t_dzv) + o, CK * HP);
        cp(SA + (l * 3 + 2) * CK * HP, INP(t_dzg) + o, CK * HP);
      }
#if HEAD
      cp(SE, INP(t_xe) + (gb * NSTEP + k0) * HP, CK * HP);
      cp(SD, INP(t_dpre) + (gb * NSTEP + k0) * HDP, CK * HDP);
#endif
    }
    __syncthreads();
    // S4D moments (s4grad.cu's recurrences)
#pragma unroll
    for (int r = 0; r < RM; r++) {
      const int m = (slice * RM + r) * NT + t;
      if (m < NM) {
        const int l = m / (HP * NSP), h = (m / NSP) % HP;
        const float* xu = SX + l * 2 * HP + h;
        for (int kk = 0; kk < CK; kk++) {
          const float u = xu[kk * WX], cy = xu[kk * WX + HP];
          const float ndr = Ar[r] * dr[r] - Ai[r] * di[r] + sr[r], ndi = Ar[r] * di[r] + Ai[r] * dr[r] + si[r];
          const float nsr = Ar[r] * sr[r] - Ai[r] * si[r] + u, nsi = Ar[r] * si[r] + Ai[r] * sr[r];
          dr[r] = ndr; di[r] = ndi; sr[r] = nsr; si[r] = nsi;
          pr[r] += cy * sr[r]; pi[r] += cy * si[r]; qr[r] += cy * dr[r]; qi[r] += cy * di[r];
        }
      }
    }
    if (mats) {
      // GLU matrices: g_Wv[i][j] += a[i] dzv[j], g_Wg[i][j] += a[i] dzg[j]; a thread owns row i, columns j0 .. j0 + 3 of both
#pragma unroll
      for (int l = 0; l < LP; l++)
#pragma unroll
        for (int r = 0; r < RW; r++) {
          const int w = t + NT * r;
          if (w < HP * HQ) {
            const int i = w / HQ, j0 = 4 * (w % HQ);
            const float* a = SA + (l * 3 + 0) * CK * HP;
            const float* zv = SA + (l * 3 + 1) * CK * HP + j0;
            const float* zg = SA + (l * 3 + 2) * CK * HP + j0;
#pragma unroll 4
            for (int kk = 0; kk < CK; kk++) {
              const float ai = a[kk * HP + i];
              const float4 v = *reinterpret_cast<const float4*>(zv + kk * HP), gg = *reinterpret_cast<const float4*>(zg + kk * HP);
              aw[l][r][0][0] = fmaf(ai, v.x, aw[l][r][0][0]); aw[l][r][0][1] = fmaf(ai, v.y, aw[l][r][0][1]);
              aw[l][r][0][2] = fmaf(ai, v.z, aw[l][r][0][2]); aw[l][r][0][3] = fmaf(ai, v.w, aw[l][r][0][3]);
              aw[l][r][1][0] = fmaf(ai, gg.x, aw[l][r][1][0]); aw[l][r][1][1] = fmaf(ai, gg.y, aw[l][r][1][1]);
              aw[l][r][1][2] = fmaf(ai, gg.z, aw[l][r][1][2]); aw[l][r][1][3] = fmaf(ai, gg.w, aw[l][r][1][3]);
            }
          }
        }
#if HEAD
#pragma unroll
      for (int r = 0; r < RH; r++) {
        const int w = t + NT * r;
        if (w < HP * (HDP / 4)) {
          const int i = w / (HDP / 4), j0 = 4 * (w % (HDP / 4));
#pragma unroll 4
          for (int kk = 0; kk < CK; kk++) {
            const float xi = SE[kk * HP + i];
            const float4 d = *reinterpret_cast<const float4*>(SD + kk * HDP + j0);
            ah[r][0] = fmaf(xi, d.x, ah[r][0]); ah[r][1] = fmaf(xi, d.y, ah[r][1]);
            ah[r][2] = fmaf(xi, d.z, ah[r][2]); ah[r][3] = fmaf(xi, d.w, ah[r][3]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < RHB; r++) {
        const int n = t + NT * r;
        if (n < HDP)
          for (int kk = 0; kk < CK; kk++) ahb[r] += SD[kk * HDP + n];
      }
#endif
      // sums: bv = sum dzv, bg = sum dzg, D = sum cy u
#pragma unroll
      for (int r = 0; r < RB; r++) {
        const int w = t + NT * r;
        if (w < 3 * LP * HP) {
          const int l = w / (3 * HP), c = (w / HP) % 3, h = w % HP;
          float s_ = 0.f;
          if (c < 2) {
            const float* z = SA + (l * 3 + 1 + c) * CK * HP + h;
            for (int kk = 0; kk < CK; kk++) s_ += z[kk * HP];
          } else {
            const float* xu = SX + l * 2 * HP + h;
            for (int kk = 0; kk < CK; kk++) s_ += xu[kk * WX + HP] * xu[kk * WX];
          }
          ab[r] += s_;
        }
      }
    }
  }
  // ---------------- outputs (per patient)
#pragma unroll
  for (int r = 0; r < RM; r++) {
    const int m = (slice * RM + r) * NT + t;
    if (m < NM) {
      float* o = OUT(pq) + (gb * NM + m) * 4;
      o[0] = pr[r]; o[1] = pi[r]; o[2] = qr[r]; o[3] = qi[r];
    }
  }
  if (!mats) return;
#pragma unroll
  for (int l = 0; l < LP; l++)
#pragma unroll
    for (int r = 0; r < RW; r++) {
      const int w = t + NT * r;
      if (w < HP * HQ) {
        const int i = w / HQ, j0 = 4 * (w % HQ);
#pragma unroll
        for (int p = 0; p < 2; p++)
          *reinterpret_cast<float4*>(OUT(g_w) + ((((gb * LP + l) * 2 + p) * HP + i) * HP + j0)) =
              make_float4(aw[l][r][p][0], aw[l][r][p][1], aw[l][r][p][2], aw[l][r][p][3]);
      }
    }
#pragma unroll
  for (int r = 0; r < RB; r++) {
    const int w = t + NT * r;
    if (w < 3 * LP * HP) OUT(g_b)[gb * 3 * LP * HP + w] = ab[r];               // (l, {bv, bg, D}, h)
  }
#if HEAD
#pragma unroll
  for (int r = 0; r < RH; r++) {
    const int w = t + NT * r;
    if (w < HP * (HDP / 4)) {
      const int i = w / (HDP / 4), j0 = 4 * (w % (HDP / 4));
      *reinterpret_cast<float4*>(OUT(g_hw) + (gb * HP + i) * HDP + j0) = make_float4(ah[r][0], ah[r][1], ah[r][2], ah[r][3]);
    }
  }
#pragma unroll
  for (int r = 0; r < RHB; r++) {
    const int n = t + NT * r;
    if (n < HDP) OUT(g_hb)[gb * HDP + n] = ahb[r];
  }
#endif
}

static size_t smem_bytes() { return sizeof(float) * CK * (WX + 3 * LP * HP + HP + HDP); }

static ffi::Error Impl(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets, int32_t B) {
  Args A;
  if ((int)args.size() != NIN || (int)rets.size() != NOUT) return ffi::Error::InvalidArgument("casm: wrong number of buffers");
  for (int i = 0; i < NIN; i++) {
    auto x = args.get<ffi::AnyBuffer>(i);
    if (x.has_error()) return ffi::Error::InvalidArgument("casm: bad input");
    A.in[i] = (const float*)x.value().untyped_data();
  }
  for (int i = 0; i < NOUT; i++) {
    auto y = rets.get<ffi::AnyBuffer>(i);
    if (y.has_error()) return ffi::Error::InvalidArgument("casm: bad output");
    A.out[i] = (float*)(*y.value()).untyped_data();
  }
  const int G = (int)(args.get<ffi::AnyBuffer>(I_dA).value().element_count() / (2 * NM));     // members (leading axis of every buffer)
  const size_t smem = smem_bytes();
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(casm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr = true;
  }
  dim3 grid(MS, B, G);
  casm_kernel<<<grid, NT, smem, stream>>>(A, B);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CasmHandler, Impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().RemainingArgs().RemainingRets().Attr<int32_t>("B"));
