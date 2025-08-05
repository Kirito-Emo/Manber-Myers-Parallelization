// Created by Emanuele (https://github.com/Kirito-Emo)

#include "suffix_array_cuda.h"
#include "cuda_kernels.cuh"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <iostream>

// Build the suffix array using Manber-Myers on GPU
void build_suffix_array_cuda(const std::vector<uint8_t>& h_text, std::vector<int>& h_sa)
{
    int n = h_text.size();
    h_sa.resize(n);

    // Device buffers
    thrust::device_vector<uint8_t> d_text = h_text;
    thrust::device_vector<int> d_sa(n), d_rank(n), d_new_rank(n), d_keys(n);

    // Initialize sa = [0, 1, 2, ..., n-1]
    thrust::sequence(d_sa.begin(), d_sa.end());

    // Initial ranks: rank[i] = text[i]
    init_ranks_from_text(thrust::raw_pointer_cast(d_text.data()), thrust::raw_pointer_cast(d_rank.data()), n);

    int k = 1;
    while (k < n)
    {
        // Compute packed (rank[i], rank[i + k]) into keys[i]
        compute_rank_pairs(thrust::raw_pointer_cast(d_rank.data()), thrust::raw_pointer_cast(d_keys.data()), n, k);

        // Sort SA by key pairs
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_sa.begin());

        // Generate new ranks from sorted key pairs
        update_ranks(thrust::raw_pointer_cast(d_sa.data()),
                     thrust::raw_pointer_cast(d_rank.data()),
                     thrust::raw_pointer_cast(d_new_rank.data()), n);

        // Swap ranks
        d_rank.swap(d_new_rank);

        // Early stop if ranks are unique
        int last_rank;
        CUDA_CHECK(cudaMemcpy(&last_rank, thrust::raw_pointer_cast(d_rank.data() + n - 1), sizeof(int), cudaMemcpyDeviceToHost));
        if (last_rank == n - 1)
            break;

        k <<= 1;
    }

    // Copy final suffix array to host
    thrust::copy(d_sa.begin(), d_sa.end(), h_sa.begin());
}

// Find the longest repeated substring on GPU using LCP
void find_lrs_cuda(const std::vector<uint8_t>& h_text, const std::vector<int>& h_sa, int& lrs_pos, int& lrs_len)
{
    int n = h_text.size();
    lrs_pos = 0;
    lrs_len = 0;

    thrust::device_vector<uint8_t> d_text = h_text;
    thrust::device_vector<int> d_sa = h_sa;
    thrust::device_vector<int> d_rank(n), d_lcp(n);

    // Compute rank[i] = inverse of sa[i]
    compute_rank_kernel<<<(n + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_sa.data()),
        thrust::raw_pointer_cast(d_rank.data()), n
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute LCP array
    compute_lcp_kernel<<<(n + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_text.data()),
        thrust::raw_pointer_cast(d_sa.data()),
        thrust::raw_pointer_cast(d_rank.data()),
        thrust::raw_pointer_cast(d_lcp.data()), n
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Find maximum LCP
    auto max_it = thrust::max_element(d_lcp.begin(), d_lcp.end());
    lrs_len = *max_it;
    int idx = max_it - d_lcp.begin();
    lrs_pos = h_sa[idx]; // Corresponding position
}

// Reset CUDA state
void cuda_suffix_array_cleanup()
{
    CUDA_CHECK(cudaDeviceReset());
}
