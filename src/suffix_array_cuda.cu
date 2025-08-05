// Created by Emanuele (https://github.com/Kirito-Emo)

#include "suffix_array_cuda.h"
#include "cuda_kernels.cuh"
#include "cuda_utils.h"
#include "cuda_malloc_utils.h"
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <iostream>

// Build the suffix array using Manber–Myers algorithm on GPU
void build_suffix_array_cuda(const std::vector<uint8_t>& h_text, std::vector<int>& h_sa, cudaMemPool_t pool)
{
    int n = h_text.size();
    h_sa.resize(n);

    cudaStream_t stream = 0;  // Default stream

    // Allocate device memory from async pool
    uint8_t* d_text      = static_cast<uint8_t*>(cuda_malloc_async(pool, n * sizeof(uint8_t), stream));
    int* d_sa            = static_cast<int*>(cuda_malloc_async(pool, n * sizeof(int), stream));
    int* d_rank          = static_cast<int*>(cuda_malloc_async(pool, n * sizeof(int), stream));
    int* d_new_rank      = static_cast<int*>(cuda_malloc_async(pool, n * sizeof(int), stream));
    int* d_keys          = static_cast<int*>(cuda_malloc_async(pool, n * sizeof(int), stream));

    // Copy input text to device
    CUDA_CHECK(cudaMemcpyAsync(d_text, h_text.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

    // Initialize suffix array and character ranks
    thrust::device_ptr<int> dev_sa(d_sa);
    thrust::sequence(dev_sa, dev_sa + n); // sa[i] = i

    init_ranks_from_text(d_text, d_rank, n);

    // Manber–Myers main loop
    int k = 1;
    while (k < n)
    {
        compute_rank_pairs(d_rank, d_keys, n, k);

        // Sort suffixes based on key pairs
        thrust::sort_by_key(thrust::device_pointer_cast(d_keys),
                            thrust::device_pointer_cast(d_keys + n),
                            thrust::device_pointer_cast(d_sa));

        // Assign new ranks
        update_ranks(d_sa, d_rank, d_new_rank, n);
        std::swap(d_rank, d_new_rank);

        // Check for convergence (all ranks unique)
        int last_rank = -1;
        CUDA_CHECK(cudaMemcpy(&last_rank, d_rank + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
        if (last_rank == n - 1)
            break;

        k <<= 1;
    }

    // Copy suffix array back to host
    CUDA_CHECK(cudaMemcpyAsync(h_sa.data(), d_sa, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free all GPU memory
    cuda_free_async(d_text, stream);
    cuda_free_async(d_sa, stream);
    cuda_free_async(d_rank, stream);
    cuda_free_async(d_new_rank, stream);
    cuda_free_async(d_keys, stream);
}

// Compute LCP and extract longest repeated substring (LRS) entirely on GPU
void find_lrs_cuda(const std::vector<uint8_t>& h_text, const std::vector<int>& h_sa,
                   int& lrs_pos, int& lrs_len, cudaMemPool_t pool)
{
    int n = h_text.size();
    lrs_pos = 0;
    lrs_len = 0;

    cudaStream_t stream = 0;

    // Allocate memory
    uint8_t* d_text = static_cast<uint8_t*>(cuda_malloc_async(pool, n * sizeof(uint8_t), stream));
    int* d_sa       = static_cast<int*>(cuda_malloc_async(pool, n * sizeof(int), stream));
    int* d_rank     = static_cast<int*>(cuda_malloc_async(pool, n * sizeof(int), stream));
    int* d_lcp      = static_cast<int*>(cuda_malloc_async(pool, n * sizeof(int), stream));

    // Transfer data to device
    CUDA_CHECK(cudaMemcpyAsync(d_text, h_text.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sa, h_sa.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Compute rank and LCP
    compute_rank_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_sa, d_rank, n);
    compute_lcp_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_text, d_sa, d_rank, d_lcp, n);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Find the max LCP value and its index
    thrust::device_ptr<int> d_lcp_ptr(d_lcp);
    auto max_it = thrust::max_element(d_lcp_ptr, d_lcp_ptr + n);
    CUDA_CHECK(cudaMemcpy(&lrs_len, max_it.get(), sizeof(int), cudaMemcpyDeviceToHost));
    int idx = max_it - d_lcp_ptr;
    lrs_pos = h_sa[idx];

    // Cleanup
    cuda_free_async(d_text, stream);
    cuda_free_async(d_sa, stream);
    cuda_free_async(d_rank, stream);
    cuda_free_async(d_lcp, stream);
}

// Reset CUDA state
void cuda_suffix_array_cleanup()
{
    CUDA_CHECK(cudaDeviceReset());
}
