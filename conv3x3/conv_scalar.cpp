#include "conv_scalar.h"

void make_kernel(const std::string& kernel, float ker[3][3]) {
    if (kernel == "box") {
        float v = 1.0f / 9.0f;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                ker[i][j] = v;
            }
        }
    }
}

void conv3x3_scalar(const std::vector<float>& input, std::vector<float>& output, const int W, const int H,
                    const float ker[3][3]) {
    int pw = W + 2;
    int ph = H + 2;
    std::vector<float> pad(pw * ph, 0.0f);
    for (int i = 0; i < H; i++) {
        std::memcpy(&pad[(i + 1) * pw] + 1, &input[i * W], W * sizeof(float));
    }
    output.resize(W * H);
    conv3x3_scalar_fast(pad, output, W, H, ker);
}

void conv3x3_scalar_fast(const std::vector<float>& pad, std::vector<float>& output, const int W, const int H,
                         const float ker[3][3]) {
    int pw = W + 2;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float s = 0.0f;
            for (int ki = 0; ki < 3; ki++) {
                for (int kj = 0; kj < 3; kj++) {
                    s += pad[(i + ki) * pw + (j + kj)] * ker[ki][kj];
                }
            }
            output[i * W + j] = s;
        }
    }
}
