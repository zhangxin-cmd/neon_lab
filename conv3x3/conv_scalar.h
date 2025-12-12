#pragma once

#include <iostream>
#include <string>
#include <vector>

void make_kernel(const std::string& kernel, float k[3][3]);

void conv3x3_scalar(const std::vector<float>& input, std::vector<float>& output, const int W, const int H,
                    const float ker[3][3]);

void conv3x3_scalar_fast(const std::vector<float>& pad, std::vector<float>& output, const int W, const int H,
                         const float ker[3][3]);
