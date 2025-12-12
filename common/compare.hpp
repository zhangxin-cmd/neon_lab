#pragma once
#include <cmath>
#include <vector>

inline double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}