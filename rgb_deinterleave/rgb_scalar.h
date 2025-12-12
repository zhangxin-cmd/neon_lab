#pragma once
#include <vector>

void rgb_deinterleave_scalar(const std::vector<uint8_t>& image, std::vector<uint8_t>& R, std::vector<uint8_t>& G,
                             std::vector<uint8_t>& B);
