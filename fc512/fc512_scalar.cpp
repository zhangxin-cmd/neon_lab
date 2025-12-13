#include "fc512_scalar.h"
void fc512_scalar(const std::vector<float>& W, const std::vector<float>& x, const std::vector<float>& b,
                  std::vector<float>& y) {
    y.resize(512);
    for (int i = 0; i < 512; ++i) {
        y[i] = b[i];
        const float* row = &W[i * 512];
        for (int j = 0; j < 512; ++j) {
            y[i] += row[j] * x[j];
        }
    }
}