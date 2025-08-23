// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>
#include <vector>

namespace cuda_sa
{
    // Build suffix array on GPU using Manber–Myers
    // Input: host text (size n)
    // Output: host suffix array (size n)
    void build_suffix_array_cuda(const uint8_t *h_text, int n, std::vector<int> &h_sa);

    // Convenience overload for vector input
    inline void build_suffix_array_cuda(const std::vector<uint8_t> &text, std::vector<int> &sa)
    {
        sa.resize(static_cast<int>(text.size()));
        build_suffix_array_cuda(text.data(), static_cast<int>(text.size()), sa);
    }
} // namespace cuda_sa
