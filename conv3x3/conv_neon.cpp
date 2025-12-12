#include "conv_neon.h"

#include <arm_neon.h>

#include <iostream>
#include <vector>

void conv3x3_neon(const std::vector<float>& input, std::vector<float>& output, const int W, const int H,
                  const float k[3][3]) {
    int pw = W + 2, ph = H + 2;
    std::vector<float> pad(pw * ph, 0.0f);

    for (int i = 0; i < H; ++i) {
        std::memcpy(&pad[(i + 1) * pw + 1], &input[i * W], W * sizeof(float));
    }

    output.resize(H * W);

    conv3x3_neon_fast(pad, output, W, H, k);
}

void conv3x3_neon_fast(const std::vector<float>& pad, std::vector<float>& output, const int W, const int H,
                       const float k[3][3]) {
    int pw = W + 2;
    float32x4_t k00 = vdupq_n_f32(k[0][0]), k01 = vdupq_n_f32(k[0][1]), k02 = vdupq_n_f32(k[0][2]);
    float32x4_t k10 = vdupq_n_f32(k[1][0]), k11 = vdupq_n_f32(k[1][1]), k12 = vdupq_n_f32(k[1][2]);
    float32x4_t k20 = vdupq_n_f32(k[2][0]), k21 = vdupq_n_f32(k[2][1]), k22 = vdupq_n_f32(k[2][2]);

    for (int i = 0; i < H; i++) {
        const float* r0 = &pad[i * pw];
        const float* r1 = &pad[(i + 1) * pw];
        const float* r2 = &pad[(i + 2) * pw];
        int j = 0;
        for (; j <= W - 8; j += 8) {
            float32x4_t s0 = vdupq_n_f32(0.0f);
            float32x4_t s1 = vdupq_n_f32(0.0f);

            float32x4_t r0_0 = vld1q_f32(r0 + j);
            float32x4_t r0_1 = vld1q_f32(r0 + j + 4);
            float32x4_t r0_2 = vld1q_f32(r0 + j + 8);

            s0 = vmlaq_f32(s0, r0_0, k00);
            s0 = vmlaq_f32(s0, vextq_f32(r0_0, r0_1, 1), k01);
            s0 = vmlaq_f32(s0, vextq_f32(r0_0, r0_1, 2), k02);

            s1 = vmlaq_f32(s1, r0_1, k00);
            s1 = vmlaq_f32(s1, vextq_f32(r0_1, r0_2, 1), k01);
            s1 = vmlaq_f32(s1, vextq_f32(r0_1, r0_2, 2), k02);

            float32x4_t r1_0 = vld1q_f32(r1 + j);
            float32x4_t r1_1 = vld1q_f32(r1 + j + 4);
            float32x4_t r1_2 = vld1q_f32(r1 + j + 8);

            s0 = vmlaq_f32(s0, r1_0, k10);
            s0 = vmlaq_f32(s0, vextq_f32(r1_0, r1_1, 1), k11);
            s0 = vmlaq_f32(s0, vextq_f32(r1_0, r1_1, 2), k12);

            s1 = vmlaq_f32(s1, r1_1, k10);
            s1 = vmlaq_f32(s1, vextq_f32(r1_1, r1_2, 1), k11);
            s1 = vmlaq_f32(s1, vextq_f32(r1_1, r1_2, 2), k12);

            float32x4_t r2_0 = vld1q_f32(r2 + j);
            float32x4_t r2_1 = vld1q_f32(r2 + j + 4);
            float32x4_t r2_2 = vld1q_f32(r2 + j + 8);

            s0 = vmlaq_f32(s0, r2_0, k20);
            s0 = vmlaq_f32(s0, vextq_f32(r2_0, r2_1, 1), k21);
            s0 = vmlaq_f32(s0, vextq_f32(r2_0, r2_1, 2), k22);

            s1 = vmlaq_f32(s1, r2_1, k20);
            s1 = vmlaq_f32(s1, vextq_f32(r2_1, r2_2, 1), k21);
            s1 = vmlaq_f32(s1, vextq_f32(r2_1, r2_2, 2), k22);

            vst1q_f32(&output[i * W + j], s0);
            vst1q_f32(&output[i * W + j + 4], s1);
        }

        for (; j <= W - 4; j += 4) {
            float32x4_t s = vdupq_n_f32(0.0f);

            float32x4_t r0_0 = vld1q_f32(r0 + j);
            float32x4_t r0_1 = vld1q_f32(r0 + j + 4);
            s = vmlaq_f32(s, r0_0, k00);
            s = vmlaq_f32(s, vextq_f32(r0_0, r0_1, 1), k01);
            s = vmlaq_f32(s, vextq_f32(r0_0, r0_1, 2), k02);

            float32x4_t r1_0 = vld1q_f32(r1 + j);
            float32x4_t r1_1 = vld1q_f32(r1 + j + 4);
            s = vmlaq_f32(s, r1_0, k10);
            s = vmlaq_f32(s, vextq_f32(r1_0, r1_1, 1), k11);
            s = vmlaq_f32(s, vextq_f32(r1_0, r1_1, 2), k12);

            float32x4_t r2_0 = vld1q_f32(r2 + j);
            float32x4_t r2_1 = vld1q_f32(r2 + j + 4);
            s = vmlaq_f32(s, r2_0, k20);
            s = vmlaq_f32(s, vextq_f32(r2_0, r2_1, 1), k21);
            s = vmlaq_f32(s, vextq_f32(r2_0, r2_1, 2), k22);

            vst1q_f32(&output[i * W + j], s);
        }
        for (; j < W; j++) {
            float s = 0.0f;
            s += r0[j + 0] * k[0][0] + r0[j + 1] * k[0][1] + r0[j + 2] * k[0][2] + r1[j + 0] * k[1][0] +
                 r1[j + 1] * k[1][1] + r1[j + 2] * k[1][2] + r2[j + 0] * k[2][0] + r2[j + 1] * k[2][1] +
                 r2[j + 2] * k[2][2];
            output[i * W + j] = s;
        }
    }
}
