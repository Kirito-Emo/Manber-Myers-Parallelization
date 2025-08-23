#pragma once
#include <cstdint>
#include <vector>

// Multi-stream CUDA Manber–Myers (in-core) with chunked radix + pairwise merges
namespace cuda_sa_ms
{
    // Build suffix array using CUDA Manber–Myers with S streams (#S preferred = 4 or 8)
    // h_text : host pointer to text bytes (size n)
    // n      : text length
    // h_sa   : output suffix array (size n)
    // streams: number of CUDA streams to use (>=1)
    void build_suffix_array_cuda_ms(const uint8_t *h_text, int n, std::vector<int> &h_sa, int streams = 4);

    // Convenience overload
    inline void build_suffix_array_cuda_ms(const std::vector<uint8_t> &text, std::vector<int> &sa, int streams = 4)
    {
        sa.resize(static_cast<int>(text.size()));
        build_suffix_array_cuda_ms(text.data(), static_cast<int>(text.size()), sa, streams);
    }
} // namespace cuda_sa_ms
