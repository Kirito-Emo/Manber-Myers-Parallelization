// Created by Emanuele (https://github.com/Kirito-Emo)

#include "suffix_array_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

// ---- CUDA error checking -----------------------------------------------------

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(code) +
                                 " at " + file + ":" + std::to_string(line));
    }
}

namespace cuda_sa
{
// ---- Kernels: Manber–Myers --------------------------------------------------

// Initialize ranks from input text bytes
__global__ void k_init_ranks(const uint8_t* text, int* rank, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        rank[i] = static_cast<int>(text[i]);
    }
}

// Build 32-bit secondary keys: key = rank[i+k]+1 (0 when i+k>=n)
__global__ void k_build_key_r2(const int* rank, uint32_t* key, int n, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        key[i] = (i + k < n) ? static_cast<uint32_t>(rank[i + k] + 1) : 0u;
    }
}

// Mark boundaries using ranks (no need to keep composite keys around)
__global__ void k_mark_groups_by_rank_u8(const int* sa, const int* rank, int n, int k,
                                         unsigned char* flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        if (i == 0)
        {
            flags[i] = 0;
            return;
        }
        int a = sa[i];
        int b = sa[i - 1];

        int r1a = rank[a];
        int r1b = rank[b];

        int r2a = (a + k < n) ? (rank[a + k] + 1) : 0;
        int r2b = (b + k < n) ? (rank[b + k] + 1) : 0;

        // Start new group if (r1,r2) changed
        flags[i] = (r1a != r1b) || (r2a != r2b);
    }
}

// Scatter ranks from sorted order back to text index order
__global__ void k_scatter_ranks(const int* sa_sorted, const int* rank_sorted, int* new_rank, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int pos = sa_sorted[i];
        if (pos >= 0 && pos < n)
        {
            new_rank[pos] = rank_sorted[i];
        }
    }
}

// ---- Validation --------------------------------------------------------------

__global__ void k_validate_sa(const int* sa, int n, unsigned char* seen,
                              int* err, int* bad_i, int* bad_val)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            int v = sa[i];
            if (v < 0 || v >= n) { *err = 101; *bad_i = i; *bad_val = v; return; }
            if (seen[v])         { *err = 102; *bad_i = i; *bad_val = v; return; }
            seen[v] = 1;
        }
        *err = 0;
    }
}

// ---- Byte->int functor for scan ---------------------------------------------

struct UCharToInt
{
    __host__ __device__ int operator()(unsigned char x) const
    {
        return x ? 1 : 0;
    }
};

// ---- Public API --------------------------------------------------------------

void build_suffix_array_cuda(const uint8_t* h_text, int n, std::vector<int>& h_sa)
{
    if (n <= 0)
    {
        h_sa.clear();
        return;
    }

    // Device buffers (lean: 32-bit keys + 1-byte flags)
    thrust::device_vector<uint8_t>       d_text(h_text, h_text + n);
    thrust::device_vector<int>           d_rank(n), d_new_rank(n);
    thrust::device_vector<int>           d_sa(n), d_rank_scan(n);
    thrust::device_vector<unsigned char> d_flags(n);
    thrust::device_vector<uint32_t>      d_key32(n);  // reused each pass

    const int BLOCK = 256;
    int gridN = (n + BLOCK - 1) / BLOCK;

    // Init ranks from text bytes
    k_init_ranks<<<gridN, BLOCK>>>(thrust::raw_pointer_cast(d_text.data()),
                                   thrust::raw_pointer_cast(d_rank.data()), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Manber–Myers doubling
    for (int k = 1; ; k <<= 1)
    {
        // Values for sort are suffix indices [0..n-1]
        thrust::sequence(d_sa.begin(), d_sa.end());

        // ---- PASS 1: stable sort by secondary key r2 = rank[i+k]+1 ----
        k_build_key_r2<<<gridN, BLOCK>>>(thrust::raw_pointer_cast(d_rank.data()),
                                         thrust::raw_pointer_cast(d_key32.data()),
                                         n, k);
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::stable_sort_by_key(d_key32.begin(), d_key32.end(), d_sa.begin());

        // ---- PASS 2: stable sort by primary key r1 = rank[i] (gather via SA) ----
        // Build keys in the order of current SA: key32[j] = rank[ SA[j] ]
        thrust::gather(d_sa.begin(), d_sa.end(), d_rank.begin(), d_key32.begin());
        thrust::stable_sort_by_key(d_key32.begin(), d_key32.end(), d_sa.begin());

        // ---- Mark new groups by comparing (r1,r2) via ranks (1-byte flags) ----
        k_mark_groups_by_rank_u8<<<gridN, BLOCK>>>(
            thrust::raw_pointer_cast(d_sa.data()),
            thrust::raw_pointer_cast(d_rank.data()),
            n, k,
            thrust::raw_pointer_cast(d_flags.data()));
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- flags(u8) -> int, exclusive_scan -> compact group IDs ----
        thrust::transform(d_flags.begin(), d_flags.end(),
                          d_rank_scan.begin(), UCharToInt{});
        thrust::exclusive_scan(d_rank_scan.begin(), d_rank_scan.end(),
                               d_rank_scan.begin());

        // ---- Scatter group IDs back to text order as new ranks ----
        k_scatter_ranks<<<gridN, BLOCK>>>(
            thrust::raw_pointer_cast(d_sa.data()),
            thrust::raw_pointer_cast(d_rank_scan.data()),
            thrust::raw_pointer_cast(d_new_rank.data()), n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check if all ranks are distinct
        int max_rank = thrust::reduce(d_new_rank.begin(), d_new_rank.end(),
                                      -1, thrust::maximum<int>());

        d_rank.swap(d_new_rank);
        if (max_rank == n - 1) break;
        if (k > n) break; // safety
    }

    // Optional: device-side validation (skip if low VRAM)
#if 1
    {
        thrust::device_vector<unsigned char> d_seen(n);
        thrust::device_vector<int> d_err(1), d_bad_i(1), d_bad_val(1);
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_seen.data()), 0, n * sizeof(unsigned char)));
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_err.data()), 0, sizeof(int)));

        k_validate_sa<<<1,1>>>(thrust::raw_pointer_cast(d_sa.data()), n,
                               thrust::raw_pointer_cast(d_seen.data()),
                               thrust::raw_pointer_cast(d_err.data()),
                               thrust::raw_pointer_cast(d_bad_i.data()),
                               thrust::raw_pointer_cast(d_bad_val.data()));
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_err=0; CUDA_CHECK(cudaMemcpy(&h_err, thrust::raw_pointer_cast(d_err.data()),
                                           sizeof(int), cudaMemcpyDeviceToHost));
        if (h_err != 0)
        {
            throw std::runtime_error("Invalid SA after CUDA Manber–Myers (32-bit two-pass).");
        }
    }
#endif

    // Copy SA back to host
    h_sa.resize(n);
    CUDA_CHECK(cudaMemcpy(h_sa.data(), thrust::raw_pointer_cast(d_sa.data()),
                          n * sizeof(int), cudaMemcpyDeviceToHost));
}

} // namespace cuda_sa
