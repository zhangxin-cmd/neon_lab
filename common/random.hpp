#pragma once
#include <random>
#include <vector>

inline void fill_f32(std::vector<float>& v, float lo = 0.0f, float hi = 1.0f, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}