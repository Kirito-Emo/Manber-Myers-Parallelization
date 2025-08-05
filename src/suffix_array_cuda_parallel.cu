// Created by Emanuele (https://github.com/Kirito-Emo)

#include "suffix_array_cuda_parallel.h"
#include "cuda_kernels_parallel.cuh"
#include "cuda_utils.h"
#include "cuda_malloc_utils.h"
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <iostream>

// Parallel suffix array build using CUDA streams
void build_suffix_array_cuda_parallel(const std::vector<uint8_t> &h_text, std::vector<int> &h_sa, int num_streams, cudaMemPool_t pool)
{
    int n = h_text.size();
    h_sa.resize(n);

    int chunk_size = (n + num_streams - 1) / num_streams;

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Allocate full-size device buffers
    uint8_t *d_text = static_cast<uint8_t *>(cuda_malloc_async(pool, n * sizeof(uint8_t)));
    int *d_sa = static_cast<int *>(cuda_malloc_async(pool, n * sizeof(int)));
    int *d_rank = static_cast<int *>(cuda_malloc_async(pool, n * sizeof(int)));
    int *d_new_rank = static_cast<int *>(cuda_malloc_async(pool, n * sizeof(int)));
    int *d_keys = static_cast<int *>(cuda_malloc_async(pool, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_text, h_text.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[0]));

    // Initialize suffix array with [0, 1, ..., n-1]
    thrust::device_ptr<int> sa_ptr(d_sa);
    thrust::sequence(sa_ptr, sa_ptr + n);

    init_ranks_from_text(d_text, d_rank, n, streams[0]); // Rank[i] = text[i]

    int k = 1;
    while (k < n)
    {
        // Launch parallel tasks on each stream
        for (int s = 0; s < num_streams; ++s)
        {
            int offset = s * chunk_size;
            int size = std::min(chunk_size, n - offset);
            if (size <= 0) continue;

            compute_rank_pairs(d_rank + offset, d_keys + offset, size, k, streams[s]);
        }

        // Wait for rank-pairs computation
        for (int s = 0; s < num_streams; ++s)
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));

        // Global sort
        thrust::stable_sort_by_key(thrust::device_pointer_cast(d_keys),
                                    thrust::device_pointer_cast(d_keys + n),
                                    thrust::device_pointer_cast(d_sa));

        // Update ranks sequentially (for now)
        update_ranks(d_sa, d_rank, d_new_rank, n, streams[0]);
        std::swap(d_rank, d_new_rank);

        // Early termination
        int last_rank = -1;
        CUDA_CHECK(cudaMemcpy(&last_rank, d_rank + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
        if (last_rank == n - 1)
            break;

        k <<= 1;
    }

    // Copy SA to host
    CUDA_CHECK(cudaMemcpy(h_sa.data(), d_sa, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Free GPU buffers
    cuda_free_async(d_text);
    cuda_free_async(d_sa);
    cuda_free_async(d_rank);
    cuda_free_async(d_new_rank);
    cuda_free_async(d_keys);

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));
}

// Same as before (single-stream LCP + max search)
void find_lrs_cuda_parallel(const std::vector<uint8_t> &h_text, const std::vector<int> &h_sa, int &lrs_pos, int &lrs_len, cudaMemPool_t pool)
{
    int n = h_text.size();
    lrs_pos = 0;
    lrs_len = 0;

    cudaStream_t stream = 0;

    uint8_t *d_text = static_cast<uint8_t *>(cuda_malloc_async(pool, n * sizeof(uint8_t)));
    int *d_sa = static_cast<int *>(cuda_malloc_async(pool, n * sizeof(int)));
    int *d_rank = static_cast<int *>(cuda_malloc_async(pool, n * sizeof(int)));
    int *d_lcp = static_cast<int *>(cuda_malloc_async(pool, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_text, h_text.data(), n * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sa, h_sa.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream));

    compute_rank_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_sa, d_rank, n);
    compute_lcp_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_text, d_sa, d_rank, d_lcp, n);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Max LCP
    thrust::device_ptr<int> lcp_ptr(d_lcp);
    auto max_it = thrust::max_element(lcp_ptr, lcp_ptr + n);
    CUDA_CHECK(cudaMemcpy(&lrs_len, max_it.get(), sizeof(int), cudaMemcpyDeviceToHost));

    int idx = max_it - lcp_ptr;
    lrs_pos = h_sa[idx];

    cuda_free_async(d_text);
    cuda_free_async(d_sa);
    cuda_free_async(d_rank);
    cuda_free_async(d_lcp);
}

// Reset CUDA state
void cuda_suffix_array_cleanup()
{
    CUDA_CHECK(cudaDeviceReset());
}
