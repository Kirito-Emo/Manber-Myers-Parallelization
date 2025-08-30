// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>
#include <vector>

namespace cuda_sa
{
    struct Metrics
    {
        // Host-side wall-clock (seconds)
        double h_alloc_s = 0.0; // allocations/initializations host + activation buffer device
        double h_h2d_s = 0.0; // total copies Host->Device
        double h_d2h_s = 0.0; // total copies Device->Host

        // GPU kernel time (seconds) measured with cudaEventElapsedTime
        double gpu_kernel_s = 0.0;
    };

    // Build suffix array on GPU using Manber–Myers (single stream) and collect metrics
    // Input: host text (size n)
    // Output: host suffix array (size n) + metrics
    void build_suffix_array_cuda(const uint8_t *h_text, int n, std::vector<int> &h_sa, Metrics &m);

    // Convenience overload per vector input
    inline void build_suffix_array_cuda(const std::vector<uint8_t> &text, std::vector<int> &sa, Metrics &m)
    {
        sa.resize(static_cast<int>(text.size()));
        build_suffix_array_cuda(text.data(), static_cast<int>(text.size()), sa, m);
    }

    // Backward-compatible overload (metrics ignored)
    inline void build_suffix_array_cuda(const uint8_t *h_text, int n, std::vector<int> &h_sa)
    {
        Metrics tmp;
        build_suffix_array_cuda(h_text, n, h_sa, tmp);
    }

    inline void build_suffix_array_cuda(const std::vector<uint8_t> &text, std::vector<int> &sa)
    {
        Metrics tmp;
        build_suffix_array_cuda(text, sa, tmp);
    }
} // namespace cuda_sa
