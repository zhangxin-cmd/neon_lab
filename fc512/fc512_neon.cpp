#include "fc512_neon.h"

#include <arm_neon.h>
void fc512_neon(const std::vector<float>& W, const std::vector<float>& x, const std::vector<float>& b,
                std::vector<float>& y) {
    y.resize(512);
    for (int i = 0; i < 512; ++i) {
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f), acc2 = vdupq_n_f32(0.0f),
                    acc3 = vdupq_n_f32(0.0f);
        const float* row = &W[i * 512];
        for (int j = 0; j < 512; j += 16) {
            float32x4_t w0 = vld1q_f32(row + j);
            float32x4_t w1 = vld1q_f32(row + j + 4);
            float32x4_t w2 = vld1q_f32(row + j + 8);
            float32x4_t w3 = vld1q_f32(row + j + 12);
            float32x4_t x0 = vld1q_f32(&x[j]);
            float32x4_t x1 = vld1q_f32(&x[j + 4]);
            float32x4_t x2 = vld1q_f32(&x[j + 8]);
            float32x4_t x3 = vld1q_f32(&x[j + 12]);
            acc0 = vfmaq_f32(acc0, w0, x0);
            acc1 = vfmaq_f32(acc1, w1, x1);
            acc2 = vfmaq_f32(acc2, w2, x2);
            acc3 = vfmaq_f32(acc3, w3, x3);
        }
        y[i] = b[i] + vaddvq_f32(acc0) + vaddvq_f32(acc1) + vaddvq_f32(acc2) + vaddvq_f32(acc3);
    }
}
