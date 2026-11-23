// ESP-DSP's plain C kernels (esp-dsp master, modules/*/float/*_ansi.c), verbatim apart from the return type: their cycle counts on real
// chips are published (ESP-DSP benchmarks, -O2), so their instruction counts here calibrate cycles per instruction
__attribute__((noinline)) int dsps_dotprod_f32_ansi(const float *src1, const float *src2, float *dest, int len)
{
    float acc = 0;
    for (int i = 0 ; i < len ; i++) {
        acc += src1[i] * src2[i];
    }
    *dest = acc;
    return 0;
}

__attribute__((noinline)) int dspm_mult_f32_ansi(const float *A, const float *B, float *C, int m, int n, int k)
{
    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < k ; j++) {
            C[i * k + j] = A[i * n] * B[j];
            for (int s = 1; s < n ; s++) {
                C[i * k + j] += A[i * n + s] * B[s * k + j];
            }
        }
    }
    return 0;
}

__attribute__((noinline)) int dsps_biquad_f32_ansi(const float *input, float *output, int len, float *coef, float *w)
{
    for (int i = 0 ; i < len ; i++) {
        float d0 = input[i] - coef[3] * w[0] - coef[4] * w[1];
        output[i] = coef[0] * d0 +  coef[1] * w[0] + coef[2] * w[1];
        w[1] = w[0];
        w[0] = d0;
    }
    return 0;
}
