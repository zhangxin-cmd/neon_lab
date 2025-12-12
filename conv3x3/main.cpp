#include <arm_neon.h>

#include <cstddef>
#include <iostream>
#include <vector>

#include "../common/compare.hpp"
#include "../common/random.hpp"
#include "../common/timer.hpp"
#include "conv_neon.h"
#include "conv_scalar.h"

volatile float sink = 0.0f;

int main() {
    int width = 1920, height = 1080, iters = 1000;
    std::string kernel = "box";

    std::vector<float> img(width * height, 0.0f);
    fill_f32(img);

    float k[3][3];
    make_kernel(kernel, k);

    std::vector<float> out_s, out_n;

    conv3x3_scalar(img, out_s, width, height, k);
    conv3x3_neon(img, out_n, width, height, k);

    double md = max_abs_diff(out_s, out_n);
    std::cout << "max abs diff: " << md << std::endl;

    for (size_t i = 0; i < out_s.size(); ++i) {
        sink += out_s[i] + out_n[i];
    }
    std::cout << "sink: " << sink << std::endl;

    int pw = width + 2;
    int ph = height + 2;
    std::vector<float> padded_input(pw * ph, 0.0f);
    for (int i = 0; i < height; ++i) {
        std::memcpy(&padded_input[(i + 1) * pw + 1], &img[i * width], width * sizeof(float));
    }
    out_s.resize(width * height);
    out_n.resize(width * height);

    auto bs = bench_loop(iters, [&]() { conv3x3_scalar_fast(padded_input, out_s, width, height, k); });

    auto bs_n = bench_loop(iters, [&]() { conv3x3_neon_fast(padded_input, out_n, width, height, k); });

    std::cout << "scalar (core): " << bs.total_seconds << "s, " << bs.fps << " fps" << std::endl;
    std::cout << "neon   (core): " << bs_n.total_seconds << "s, " << bs_n.fps << " fps" << std::endl;
    return 0;
}
