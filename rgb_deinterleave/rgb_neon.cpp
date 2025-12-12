#include "rgb_neon.h"

#include <arm_neon.h>

void rgb_deinterleave_neon(const std::vector<uint8_t>& image, std::vector<uint8_t>& R, std::vector<uint8_t>& G,
                           std::vector<uint8_t>& B) {
    size_t n = image.size() / 3;
    R.resize(n);
    G.resize(n);
    B.resize(n);
    size_t i = 0;
    for (; i <= n - 16; i += 16) {
        uint8x16x3_t rgb = vld3q_u8(&image[i * 3]);
        vst1q_u8(&R[i], rgb.val[0]);
        vst1q_u8(&G[i], rgb.val[1]);
        vst1q_u8(&B[i], rgb.val[2]);
    }
    for (; i < n; ++i) {
        R[i] = image[3 * i + 0];
        G[i] = image[3 * i + 1];
        B[i] = image[3 * i + 2];
    }
}