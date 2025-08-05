// Created by Emanuele (https://github.com/Kirito-Emo)

#include "cuda_kernels_parallel.cuh"
#include "cuda_utils.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define BLOCK_SIZE 256

// Kernel to initialize ranks from text: rank[i] = text[i] (stream-safe)
__global__ void init_ranks_kernel(const uint8_t* __restrict__ text, int* __restrict__ rank, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        rank[i] = static_cast<int>(text[i]);
}

// Host function: initialize ranks on given CUDA stream
void init_ranks_from_text(const uint8_t* d_text, int* d_rank, int n, cudaStream_t stream) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_ranks_kernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_text, d_rank, n);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to pack (rank[i], rank[i+k]) into 32-bit key for radix sorting
__global__ void compute_rank_pairs_kernel(const int* __restrict__ rank, int* __restrict__ keys, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int high = rank[i];
        int low  = (i + k < n) ? rank[i + k] : -1;
        keys[i]  = (high << 16) | (low & 0xFFFF);
    }
}

// Host wrapper: compute key pairs asynchronously on stream
void compute_rank_pairs(const int* d_rank, int* d_keys, int n, int k, cudaStream_t stream) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_rank_pairs_kernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_rank, d_keys, n, k);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to assign "unique change" flags into new_rank[] based on key comparison
__global__ void update_ranks_kernel(const int* __restrict__ sa,
                                    const int* __restrict__ old_rank,
                                    int* __restrict__ new_rank,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (i == 0) {
        new_rank[sa[0]] = 0;
        return;
    }

    int curr = sa[i];
    int prev = sa[i - 1];

    int curr_hi = old_rank[curr];
    int curr_lo = (curr + 1 < n) ? old_rank[curr + 1] : -1;

    int prev_hi = old_rank[prev];
    int prev_lo = (prev + 1 < n) ? old_rank[prev + 1] : -1;

    new_rank[curr] = (curr_hi != prev_hi || curr_lo != prev_lo) ? 1 : 0;
}

// Full update of ranks including parallel inclusive scan
void update_ranks(const int* d_sa, const int* d_rank, int* d_new_rank, int n, cudaStream_t stream) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_ranks_kernel<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_sa, d_rank, d_new_rank, n);
    CUDA_CHECK(cudaGetLastError());

    // Parallel prefix sum to assign global ranks
    auto new_rank_ptr = thrust::device_pointer_cast(d_new_rank);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), new_rank_ptr, new_rank_ptr + n, new_rank_ptr);
}

// Compute inverse suffix array: rank[sa[i]] = i
__global__ void compute_rank_kernel(const int* __restrict__ sa, int* __restrict__ rank, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        rank[sa[i]] = i;
}

// Compute LCP array using Kasai algorithm with shared memory
__global__ void compute_lcp_kernel(const uint8_t* __restrict__ text,
                                   const int* __restrict__ sa,
                                   const int* __restrict__ rank,
                                   int* __restrict__ lcp,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    __shared__ uint8_t sh_text[BLOCK_SIZE];
    __shared__ int sh_sa[BLOCK_SIZE];
    __shared__ int sh_rank[BLOCK_SIZE];

    int tid = threadIdx.x;
    if (i < n) {
        sh_text[tid] = text[i];
        sh_sa[tid] = sa[i];
        sh_rank[tid] = rank[i];
    }
    __syncthreads();

    if (sh_rank[tid] == 0) {
        lcp[i] = 0;
        return;
    }

    int j = sa[sh_rank[tid] - 1];
    int len = 0;
    while (i + len < n && j + len < n && text[i + len] == text[j + len])
        ++len;

    lcp[i] = len;
}
