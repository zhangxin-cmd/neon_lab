#pragma once
#include <vector>

void fc512_neon(const std::vector<float>& W, const std::vector<float>& x, const std::vector<float>& b,
                std::vector<float>& y);