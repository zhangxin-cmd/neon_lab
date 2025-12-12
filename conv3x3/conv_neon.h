#pragma once
#include <vector>

void conv3x3_neon(const std::vector<float>& img, std::vector<float>& out, const int width, const int height,
                  const float k[3][3]);

void conv3x3_neon_fast(const std::vector<float>& padded_input, std::vector<float>& out, const int width,
                       const int height, const float k[3][3]);
