// Created by Emanuele (https://github.com/Kirito-Emo)

#include "cuda_kernels.cuh"
#include <cuda_runtime.h>

// Compute inverse suffix array (rank[i] = position of i in SA)
__global__ void init_ranks_kernel(const uint8_t* __restrict__ text, int* __restrict__ rank, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        rank[i] = static_cast<int>(text[i]);
}

// Initialize ranks from input text characters (1-byte) - Each rank[i] = text[i] (0-255)
void init_ranks_from_text(const uint8_t* d_text, int* d_rank, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_ranks_kernel<<<gridSize, BLOCK_SIZE>>>(d_text, d_rank, n);
    CUDA_CHECK(cudaGetLastError());
}

// Compute (rank[i], rank[i + k]) pairs → pack into keys[i]
// Used for sorting suffixes by 2*k prefix during Manber–Myers steps
__global__ void compute_rank_pairs_kernel(const int* __restrict__ rank, int* __restrict__ keys, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int high = rank[i];
        int low = (i + k < n) ? rank[i + k] : -1;
        keys[i] = (high << 16) | (low & 0xFFFF); // Compact 2x16-bit ranks into a 32-bit key
    }
}

// Compute rank pairs (rank[i], rank[i + k]) and store in keys
// This is used for sorting suffixes by 2*k prefix during Manber–Myers
void compute_rank_pairs(const int* d_rank, int* d_keys, int n, int k)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_rank_pairs_kernel<<<gridSize, BLOCK_SIZE>>>(d_rank, d_keys, n, k);
    CUDA_CHECK(cudaGetLastError());
}

// Generate new ranks after sorting suffixes by key pairs
__global__ void update_ranks_kernel(const int* __restrict__ sa, const int* __restrict__ old_rank, int* __restrict__ new_rank, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (i == 0)
    {
        new_rank[sa[0]] = 0;
        return;
    }

    int curr = sa[i];
    int prev = sa[i - 1];

    int curr_hi = old_rank[curr];
    int curr_lo = (curr + 1 < n) ? old_rank[curr + 1] : -1;

    int prev_hi = old_rank[prev];
    int prev_lo = (prev + 1 < n) ? old_rank[prev + 1] : -1;

    int is_diff = (curr_hi != prev_hi || curr_lo != prev_lo);
    new_rank[curr] = is_diff;
}

// Prefix sum to finalize ranks (sequential prefix sum kernel)
__global__ void prefix_sum_kernel(int* data, int n)
{
    // NOTE: This kernel is sequential and runs on a single thread
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 1; i < n; ++i)
            data[i] += data[i - 1];
    }
}

// Update ranks after sorting suffixes by keys
void update_ranks(const int* d_sa, const int* d_rank, int* d_new_rank, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_ranks_kernel<<<gridSize, BLOCK_SIZE>>>(d_sa, d_rank, d_new_rank, n);
    CUDA_CHECK(cudaGetLastError());

    prefix_sum_kernel<<<1, 1>>>(d_new_rank, n); // Finalize ranks with inclusive scan
    CUDA_CHECK(cudaGetLastError());
}

// Compute inverse suffix array: rank[sa[i]] = i
__global__ void compute_rank_kernel(const int* __restrict__ sa, int* __restrict__ rank, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        rank[sa[i]] = i;
}

// Optimized Kasai algorithm: computes the LCP array on GPU
__global__ void compute_lcp_kernel(const uint8_t* __restrict__ text, const int* __restrict__ sa, const int* __restrict__ rank, int* __restrict__ lcp, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Shared memory per block — assume blockDim.x <= 1024
    __shared__ uint8_t sh_text[BLOCK_SIZE];
    __shared__ int sh_sa[BLOCK_SIZE];
    __shared__ int sh_rank[BLOCK_SIZE];

    int tid = threadIdx.x;
    if (i < n)
    {
        sh_text[tid] = text[i];
        sh_sa[tid] = sa[i];
        sh_rank[tid] = rank[i];
    }
    __syncthreads();

    if (sh_rank[tid] == 0)
    {
        lcp[i] = 0;
        return;
    }

    int j = sa[sh_rank[tid] - 1];
    int len = 0;

    // Compare characters starting from position i and j
    while ((i + len < n) && (j + len < n) && text[i + len] == text[j + len])
        ++len;

    lcp[i] = len;
}
