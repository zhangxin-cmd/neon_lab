#include <iostream>
#include <vector>

#include "../common/compare.hpp"
#include "../common/random.hpp"
#include "../common/timer.hpp"
#include "fc512_neon.h"
#include "fc512_scalar.h"

int main() {
    int iters = 500;
    std::vector<float> W(512 * 512);
    std::vector<float> x(512);
    std::vector<float> b(512);

    fill_f32(W);
    fill_f32(x);
    fill_f32(b);

    std::vector<float> ys(512);
    std::vector<float> yn(512);
    fc512_scalar(W, x, b, ys);
    fc512_neon(W, x, b, yn);
    double md = max_abs_diff(ys, yn);
    std::cout << "max abs diff: " << md << std::endl;

    auto bs = bench_loop(iters, [&]() { fc512_scalar(W, x, b, ys); });
    auto bn = bench_loop(iters, [&]() { fc512_neon(W, x, b, yn); });

    std::cout << "scalar time: " << bs.total_seconds << " s. fps: " << bs.fps << std::endl;
    std::cout << "neon time: " << bn.total_seconds << " s. fps: " << bn.fps << std::endl;
    return 0;
}