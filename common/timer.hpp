#pragma once
#include <chrono>

struct BenchResult {
    double total_seconds;
    double fps;
};

template <typename F>
inline BenchResult bench_loop(int iters, F&& fn) {
    fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d = t1 - t0;
    double total = d.count();
    double fps = (total > 0.0) ? (iters / total) : 0.0;
    return {total, fps};
}