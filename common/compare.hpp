#pragma once
#include <cmath>
#include <vector>

template <typename T>
inline double max_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
    double m = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}