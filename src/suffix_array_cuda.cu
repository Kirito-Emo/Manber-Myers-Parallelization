// Created by Emanuele (https://github.com/Kirito-Emo)

#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include "suffix_array_cuda.h"

// ---- CUDA error checking -----------------------------------------------------

#define CUDA_CHECK(ans)                                                                                                \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(code) + " at " + file + ":" +
                                 std::to_string(line));
    }
}

namespace cuda_sa
{
    // ---- Kernels: Manber–Myers --------------------------------------------------

    // Initialize ranks from input text bytes
    __global__ void k_init_ranks(const uint8_t *text, int *rank, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            rank[i] = static_cast<int>(text[i]);
    }

    // Build 32-bit secondary keys: key = rank[i+k]+1 (0 when i+k>=n)
    __global__ void k_build_key_r2(const int *rank, uint32_t *key, int n, int k)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            key[i] = (i + k < n) ? static_cast<uint32_t>(rank[i + k] + 1) : 0u;
    }

    // Mark boundaries using ranks (no need to keep composite keys around)
    __global__ void k_mark_groups_by_rank_u8(const int *sa, const int *rank, int n, int k, unsigned char *flags)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            if (i == 0)
            {
                flags[i] = 0;
                return;
            }
            int a = sa[i], b = sa[i - 1];
            int r1a = rank[a], r1b = rank[b];
            int r2a = (a + k < n) ? (rank[a + k] + 1) : 0;
            int r2b = (b + k < n) ? (rank[b + k] + 1) : 0;
            flags[i] = (r1a != r1b) || (r2a != r2b);
        }
    }

    // Scatter ranks from sorted order back to text index order
    __global__ void k_scatter_ranks(const int *sa_sorted, const int *rank_sorted, int *new_rank, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            int pos = sa_sorted[i];
            if (pos >= 0 && pos < n)
                new_rank[pos] = rank_sorted[i];
        }
    }

    // ---- Byte->int functor for scan ---------------------------------------------
    struct UCharToInt
    {
        __host__ __device__ int operator()(unsigned char x) const { return x ? 1 : 0; }
    };

    // ---- Public API --------------------------------------------------------------
    void build_suffix_array_cuda(const uint8_t *h_text, int n, std::vector<int> &h_sa, Metrics &m)
    {
        if (n <= 0)
        {
            h_sa.clear();
            m = {};
            return;
        }

        auto host_t0 = std::chrono::high_resolution_clock::now();

        // Device buffers
        thrust::device_vector<uint8_t> d_text(n);
        thrust::device_vector<int> d_rank(n), d_new_rank(n);
        thrust::device_vector<int> d_sa(n), d_rank_scan(n);
        thrust::device_vector<unsigned char> d_flags(n);
        thrust::device_vector<uint32_t> d_key32(n);

        auto host_t1 = std::chrono::high_resolution_clock::now();
        m.h_alloc_s = std::chrono::duration<double>(host_t1 - host_t0).count();

        // H->D copy (time on host)
        host_t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_text.data()), h_text, n * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        host_t1 = std::chrono::high_resolution_clock::now();
        m.h_h2d_s = std::chrono::duration<double>(host_t1 - host_t0).count();

        const int BLOCK = 256;
        int gridN = (n + BLOCK - 1) / BLOCK;

        // GPU kernel (total) timing with cudaEvent
        cudaEvent_t evStart, evStop;
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evStop));
        CUDA_CHECK(cudaEventRecord(evStart, 0));

        // Init ranks from text bytes
        k_init_ranks<<<gridN, BLOCK>>>(thrust::raw_pointer_cast(d_text.data()), thrust::raw_pointer_cast(d_rank.data()),
                                       n);
        CUDA_CHECK(cudaGetLastError());

        // Manber–Myers doubling loop (uses kernels + Thrust kernels)
        for (int k = 1;; k <<= 1)
        {
            // Values for sort are suffix indices [0..n-1]
            thrust::sequence(d_sa.begin(), d_sa.end());

            // Stable sort by secondary key r2 = rank[i+k]+1
            k_build_key_r2<<<gridN, BLOCK>>>(thrust::raw_pointer_cast(d_rank.data()),
                                             thrust::raw_pointer_cast(d_key32.data()), n, k);
            CUDA_CHECK(cudaGetLastError());
            thrust::stable_sort_by_key(d_key32.begin(), d_key32.end(), d_sa.begin());

            // Stable sort by primary key r1 = rank[i] (gather via SA)
            // Build keys in the order of current SA: key32[j] = rank[ SA[j] ]
            thrust::gather(d_sa.begin(), d_sa.end(), d_rank.begin(), d_key32.begin());
            thrust::stable_sort_by_key(d_key32.begin(), d_key32.end(), d_sa.begin());

            // Mark new groups by comparing (r1,r2) via ranks (1-byte flags)
            k_mark_groups_by_rank_u8<<<gridN, BLOCK>>>(thrust::raw_pointer_cast(d_sa.data()),
                                                       thrust::raw_pointer_cast(d_rank.data()), n, k,
                                                       thrust::raw_pointer_cast(d_flags.data()));
            CUDA_CHECK(cudaGetLastError());

            // flags(u8) -> int, exclusive_scan -> compact group IDs
            thrust::transform(d_flags.begin(), d_flags.end(), d_rank_scan.begin(), UCharToInt{});
            thrust::exclusive_scan(d_rank_scan.begin(), d_rank_scan.end(), d_rank_scan.begin());

            // Scatter group IDs back to text order as new ranks
            k_scatter_ranks<<<gridN, BLOCK>>>(thrust::raw_pointer_cast(d_sa.data()),
                                              thrust::raw_pointer_cast(d_rank_scan.data()),
                                              thrust::raw_pointer_cast(d_new_rank.data()), n);
            CUDA_CHECK(cudaGetLastError());

            // Check if all ranks are distinct
            int max_rank = thrust::reduce(d_new_rank.begin(), d_new_rank.end(), -1, thrust::maximum<int>());
            d_rank.swap(d_new_rank);
            if (max_rank == n - 1)
                break;
            if (k > n)
                break; // safety
        }

        CUDA_CHECK(cudaEventRecord(evStop, 0));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        m.gpu_kernel_s = static_cast<double>(ms) / 1000.0;

        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));

        // D->H copy of SA
        host_t0 = std::chrono::high_resolution_clock::now();
        h_sa.resize(n);
        CUDA_CHECK(cudaMemcpy(h_sa.data(), thrust::raw_pointer_cast(d_sa.data()), n * sizeof(int),
                              cudaMemcpyDeviceToHost));
        host_t1 = std::chrono::high_resolution_clock::now();
        m.h_d2h_s = std::chrono::duration<double>(host_t1 - host_t0).count();
    }

} // namespace cuda_sa
