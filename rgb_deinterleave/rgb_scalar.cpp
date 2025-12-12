#include "rgb_scalar.h"

#include <cstddef>

void rgb_deinterleave_scalar(const std::vector<uint8_t>& image, std::vector<uint8_t>& R, std::vector<uint8_t>& G,
                             std::vector<uint8_t>& B) {
    size_t n = image.size() / 3;
    R.resize(n);
    G.resize(n);
    B.resize(n);

    for (size_t i = 0; i < n; i++) {
        R[i] = image[i * 3];
        G[i] = image[i * 3 + 1];
        B[i] = image[i * 3 + 2];
    }
}
