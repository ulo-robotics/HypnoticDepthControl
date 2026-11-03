// CUDA S4D-kernel gradient: for every (member, patient, layer, channel, mode) the z-transform of the kernel cotangent at the mode's pole and its
// derivative, P = sum_m gK[m] dA^m and P' = sum_m m gK[m] dA^(m-1), where gK[m] = sum_k cy[k] u[k - m] is the cross-correlation of the S4D output
// cotangent cy with the S4D input u. Computed without the cross-correlation, by two recurrences over the steps (one thread per (channel, mode)):
//   s_k = dA s_(k-1) + u_k,  s'_k = dA s'_(k-1) + s_(k-1),  P = sum_k cy_k s_k,  P' = sum_k cy_k s'_k.
// kstep.py turns (P, P') into the gradients of log_dt, log_A_re, A_im and C. Sizes are compile-time constants (-DH=.. -DNS=.. -DNL=.. -DCG=..).
// Layouts: dA (M, NL, H, NS, 2) planes; X (M, B, N, W) holds u and cy of layer l at columns OU + l * LSTRIDE and OC + l * LSTRIDE (runtime);
//          out (M, B, NL, H, NS, 4): Re P, Im P, Re P', Im P'.
#include <cuda_runtime.h>
#include <cstdint>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#ifndef H
#error "compile with -DH=.. -DNS=.. -DNL=.. -DCG=.."
#endif
#ifndef NSTEP
#define NSTEP 360
#endif
static_assert(H % CG == 0, "CG must divide H");

__global__ void __launch_bounds__(CG * NS) s4grad_kernel(const float* __restrict__ dA, const float* __restrict__ X, float* __restrict__ out, int B, int W,
                                                         int OU, int OC, int LSTRIDE) {
  const int m = blockIdx.z, l = blockIdx.y % NL, cg = blockIdx.y / NL, b = blockIdx.x;
  const int c = cg * CG + threadIdx.x / NS, n = threadIdx.x % NS;
  const float* a = dA + ((((size_t)m * NL + l) * H + c) * NS + n) * 2;
  const float ar = a[0], ai = a[1];
  const float* x = X + ((size_t)m * B + b) * NSTEP * W;
  const int ou = OU + l * LSTRIDE + c, oc = OC + l * LSTRIDE + c;
  float sr = 0.0f, si = 0.0f, dr = 0.0f, di = 0.0f, pr = 0.0f, pi = 0.0f, qr = 0.0f, qi = 0.0f;
  for (int k = 0; k < NSTEP; k++) {
    const float u = __ldg(x + (size_t)k * W + ou), cy = __ldg(x + (size_t)k * W + oc);
    const float ndr = ar * dr - ai * di + sr, ndi = ar * di + ai * dr + si;
    const float nsr = ar * sr - ai * si + u, nsi = ar * si + ai * sr;
    dr = ndr; di = ndi; sr = nsr; si = nsi;
    pr += cy * sr; pi += cy * si; qr += cy * dr; qi += cy * di;
  }
  float* o = out + (((((size_t)m * B + b) * NL + l) * H + c) * NS + n) * 4;
  o[0] = pr; o[1] = pi; o[2] = qr; o[3] = qi;
}

static ffi::Error S4grad(cudaStream_t stream, ffi::Buffer<ffi::F32> dA, ffi::Buffer<ffi::F32> X, ffi::ResultBuffer<ffi::F32> out, int32_t ou,
                         int32_t oc, int32_t lstride) {
  const auto xd = X.dimensions();
  const int W = (int)xd[xd.size() - 1], B = (int)xd[xd.size() - 3];
  const int M = (int)(dA.element_count() / ((size_t)NL * H * NS * 2));
  dim3 grid(B, NL * (H / CG), M);
  s4grad_kernel<<<grid, CG * NS, 0, stream>>>(dA.typed_data(), X.typed_data(), out->typed_data(), B, W, ou, oc, lstride);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PkpdS4grad, S4grad,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>().Attr<int32_t>("ou").Attr<int32_t>("oc").Attr<int32_t>("lstride"));
