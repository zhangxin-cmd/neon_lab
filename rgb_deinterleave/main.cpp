#include <iostream>
#include <vector>

#include "../common/compare.hpp"
#include "../common/random.hpp"
#include "../common/timer.hpp"
#include "rgb_neon.h"
#include "rgb_scalar.h"

int main() {
    int width = 1920, height = 1080, iters = 50;
    std::vector<uint8_t> image(width * height * 3);
    fill_u8(image);

    std::vector<uint8_t> R_n, G_n, B_n;
    std::vector<uint8_t> R_s, G_s, B_s;

    rgb_deinterleave_neon(image, R_n, G_n, B_n);
    rgb_deinterleave_scalar(image, R_s, G_s, B_s);

    double max_diff_R = max_abs_diff(R_n, R_s);
    double max_diff_G = max_abs_diff(G_n, G_s);
    double max_diff_B = max_abs_diff(B_n, B_s);

    std::cout << "Max diff R: " << max_diff_R << std::endl;
    std::cout << "Max diff G: " << max_diff_G << std::endl;
    std::cout << "Max diff B: " << max_diff_B << std::endl;

    auto s = bench_loop(iters, [&]() { rgb_deinterleave_scalar(image, R_s, G_s, B_s); });
    auto v = bench_loop(iters, [&]() { rgb_deinterleave_neon(image, R_n, G_n, B_n); });

    std::cout << "scalar (core): " << s.total_seconds << "s, " << s.fps << " fps" << std::endl;
    std::cout << "neon   (core): " << v.total_seconds << "s, " << v.fps << " fps" << std::endl;
    return 0;
}
