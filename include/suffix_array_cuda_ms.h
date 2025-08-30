// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>
#include <vector>

// Multi-stream CUDA Manber–Myers (in-core) with chunked radix + pairwise merges
namespace cuda_sa_ms
{
    struct MetricsMS
    {
        // Host-side wall-clock (seconds)
        double h_alloc_s = 0.0; // allocations/initializations host + device
        double h_h2d_s = 0.0; // total copies Host->Device
        double h_d2h_s = 0.0; // total copies Device->Host

        // GPU kernel time (seconds) measured with cudaEvent
        double gpu_kernel_s = 0.0;

        int streams_used = 0; // number of CUDA streams used
    };

    // Build suffix array using CUDA Manber–Myers with S streams (#S preferred = 4 or 8)
    // h_text : host pointer to text bytes (size n)
    // n      : text length
    // h_sa   : output suffix array (size n)
    // m      : output metrics
    // streams: number of CUDA streams to use (>=1)
    void build_suffix_array_cuda_ms(const uint8_t *h_text, int n, std::vector<int> &h_sa, int streams, MetricsMS &m);


    // Convenience overload
    inline void build_suffix_array_cuda_ms(const std::vector<uint8_t> &text, std::vector<int> &sa, int streams,
                                           MetricsMS &m)
    {
        sa.resize(static_cast<int>(text.size()));
        build_suffix_array_cuda_ms(text.data(), static_cast<int>(text.size()), sa, streams, m);
    }

    // Backward-compatible overload (metrics ignored)
    inline void build_suffix_array_cuda_ms(const uint8_t *h_text, int n, std::vector<int> &h_sa, int streams = 4)
    {
        MetricsMS tmp{};
        build_suffix_array_cuda_ms(h_text, n, h_sa, streams, tmp);
    }

    inline void build_suffix_array_cuda_ms(const std::vector<uint8_t> &text, std::vector<int> &sa, int streams = 4)
    {
        MetricsMS tmp{};
        build_suffix_array_cuda_ms(text, sa, streams, tmp);
    }
} // namespace cuda_sa_ms
