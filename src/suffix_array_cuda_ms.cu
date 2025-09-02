// Created by Emanuele (https://github.com/Kirito-Emo)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <vector>
#include "suffix_array_cuda_ms.h"

// ---- CUDA error checking -----------------------------------------------------
#define CUDA_CHECK(ans)                                                                                                \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(code) + " at " + file + ":" +
                                 std::to_string(line));
}

namespace cuda_sa_ms
{

    struct UCharToInt
    {
        __host__ __device__ int operator()(unsigned char x) const { return x ? 1 : 0; }
    };

    // Comparator on SA indices using (rank[i], rank[i+k])
    struct SAKeyLess
    {
        const int *rank; // device pointer
        int n;
        int k;

        __host__ __device__ bool operator()(int a, int b) const
        {
            int r1a = rank[a], r1b = rank[b];
            if (r1a != r1b)
                return r1a < r1b;
            int r2a = (a + k < n) ? (rank[a + k] + 1) : 0;
            int r2b = (b + k < n) ? (rank[b + k] + 1) : 0;
            return r2a < r2b;
        }
    };

    // Kernels

    // Initialize ranks from text bytes
    __global__ void k_init_ranks(const uint8_t *text, int *rank, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            rank[i] = static_cast<int>(text[i]);
    }

    // Build 32-bit secondary keys on subrange: key = rank[i+k] + 1 (0 if out-of-bounds)
    __global__ void k_build_key_r2_range(const int *rank, uint32_t *key, int n, int k, int start, int len)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t < len)
        {
            int i = start + t;
            key[i] = (i + k < n) ? static_cast<uint32_t>(rank[i + k] + 1) : 0u;
        }
    }

    // Mark group boundaries by comparing (r1,r2) via ranks (1-byte flags)
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

    // Scatter group IDs (in sorted order) back to text index order
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

    // Validate SA = permutation of [0..n-1]
    __global__ void k_validate_sa(const int *sa, int n, unsigned char *seen, int *err, int *bad_i, int *bad_val)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            for (int i = 0; i < n; ++i)
            {
                int v = sa[i];

                if (v < 0 || v >= n)
                {
                    *err = 101, *bad_i = i, *bad_val = v;
                    return;
                }

                if (seen[v])
                {
                    *err = 102, *bad_i = i, *bad_val = v;
                    return;
                }

                seen[v] = 1;
            }

            *err = 0;
        }
    }

    // Helpers

    static inline void compute_chunks(int n, int S, std::vector<int> &offs, std::vector<int> &lens)
    {
        offs.resize(S);
        lens.resize(S);
        int base = n / S, rem = n % S, acc = 0;
        for (int s = 0; s < S; ++s)
        {
            int len = base + (s < rem ? 1 : 0);
            offs[s] = acc;
            lens[s] = len;
            acc += len;
        }
    }

    // Public API
    void build_suffix_array_cuda_ms(const uint8_t *h_text, int n, std::vector<int> &h_sa, int streams, MetricsMS &m)
    {
        if (n <= 0)
        {
            h_sa.clear();
            m = {};
            return;
        }

        if (streams <= 0)
            streams = 1;

        m.streams_used = streams;

        // Allocations device + H->D (host time)
        auto t_alloc0 = std::chrono::high_resolution_clock::now();

        // Device buffers
        thrust::device_vector<uint8_t> d_text(n);
        thrust::device_vector<int> d_rank(n), d_new_rank(n), d_rank_scan(n);
        thrust::device_vector<unsigned char> d_flags(n); // 1-byte flags
        thrust::device_vector<uint32_t> d_key32(n); // 32-bit keys for stable sort
        thrust::device_vector<int> d_sa_A(n); // Suffix Array (ping-pong)
        auto t_alloc1 = std::chrono::high_resolution_clock::now();
        m.h_alloc_s = std::chrono::duration<double>(t_alloc1 - t_alloc0).count();

        // Copy H2D
        auto t_h2d0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_text.data()), h_text, n * sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
        auto t_h2d1 = std::chrono::high_resolution_clock::now();
        m.h_h2d_s = std::chrono::duration<double>(t_h2d1 - t_h2d0).count();

        const int BLOCK = 256;
        int gridN = (n + BLOCK - 1) / BLOCK;

        // Pure GPU events time
        cudaEvent_t evStart, evStop;
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evStop));

        // Init ranks from text
        k_init_ranks<<<gridN, BLOCK>>>(thrust::raw_pointer_cast(d_text.data()), thrust::raw_pointer_cast(d_rank.data()),
                                       n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Create streams
        std::vector<cudaStream_t> st(streams);
        for (int s = 0; s < streams; ++s)
            CUDA_CHECK(cudaStreamCreateWithFlags(&st[s], cudaStreamNonBlocking));

        // Chunk layout
        std::vector<int> offs, lens;
        compute_chunks(n, streams, offs, lens);

        // Start GPU measurement (after init ranks to exclude H->D & prime sync)
        CUDA_CHECK(cudaEventRecord(evStart, 0));

        // Doubling loop
        for (int k = 1;; k <<= 1)
        {
            // Init SA with [0..n-1]
            thrust::sequence(d_sa_A.begin(), d_sa_A.end());

            // Per-chunk two-pass stable sorting (r2 then r1)
            for (int s = 0; s < streams; ++s)
            {
                int off = offs[s], len = lens[s], grid = (len + BLOCK - 1) / BLOCK;
                // Build r2 keys on subrange
                k_build_key_r2_range<<<grid, BLOCK, 0, st[s]>>>(thrust::raw_pointer_cast(d_rank.data()),
                                                                thrust::raw_pointer_cast(d_key32.data()), n, k, off,
                                                                len);
            }

            for (int s = 0; s < streams; ++s)
                CUDA_CHECK(cudaStreamSynchronize(st[s]));

            // Stable sort by r2: (key32, sa_A) in each chunk
            for (int s = 0; s < streams; ++s)
            {
                int off = offs[s], len = lens[s];
                auto pol = thrust::cuda::par.on(st[s]);
                thrust::stable_sort_by_key(pol, d_key32.begin() + off, d_key32.begin() + off + len,
                                           d_sa_A.begin() + off);
            }

            for (int s = 0; s < streams; ++s)
                CUDA_CHECK(cudaStreamSynchronize(st[s]));

            // Build r1 keys in SA order for each chunk, then stable sort by r1
            for (int s = 0; s < streams; ++s)
            {
                int off = offs[s], len = lens[s];
                auto pol = thrust::cuda::par.on(st[s]);

                // key32[j] = rank[ SA_A[j] ] on subrange
                thrust::gather(pol, d_sa_A.begin() + off, d_sa_A.begin() + off + len, d_rank.begin(),
                               d_key32.begin() + off);

                thrust::stable_sort_by_key(pol, d_key32.begin() + off, d_key32.begin() + off + len,
                                           d_sa_A.begin() + off);
            }
            for (int s = 0; s < streams; ++s)
                CUDA_CHECK(cudaStreamSynchronize(st[s]));

            // Global stable sort ensures correctness
            {
                auto pol = thrust::cuda::par.on(st[0]);
                thrust::stable_sort(pol, d_sa_A.begin(), d_sa_A.end(),
                                    SAKeyLess{thrust::raw_pointer_cast(d_rank.data()), n, k});
            }

            const int *sa_sorted = thrust::raw_pointer_cast(d_sa_A.data());

            // Mark flags, scan, scatter to new ranks
            k_mark_groups_by_rank_u8<<<gridN, BLOCK>>>(sa_sorted, thrust::raw_pointer_cast(d_rank.data()), n, k,
                                                       thrust::raw_pointer_cast(d_flags.data()));
            CUDA_CHECK(cudaDeviceSynchronize());

            thrust::transform(d_flags.begin(), d_flags.end(), d_rank_scan.begin(), UCharToInt{});
            thrust::exclusive_scan(d_rank_scan.begin(), d_rank_scan.end(), d_rank_scan.begin());

            k_scatter_ranks<<<gridN, BLOCK>>>(sa_sorted, thrust::raw_pointer_cast(d_rank_scan.data()),
                                              thrust::raw_pointer_cast(d_new_rank.data()), n);
            CUDA_CHECK(cudaDeviceSynchronize());

            int max_rank = thrust::reduce(d_new_rank.begin(), d_new_rank.end(), -1, thrust::maximum<int>());

            d_rank.swap(d_new_rank);
            if (max_rank == n - 1 || k > n)
                break;
        }

        // Stop measuring GPU
        CUDA_CHECK(cudaEventRecord(evStop, 0));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
        m.gpu_kernel_s = static_cast<double>(ms) / 1000.0;
        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));

        // Validation + D->H (host time)
        auto t_d2h0 = std::chrono::high_resolution_clock::now();
        const int *sa_dev = thrust::raw_pointer_cast(d_sa_A.data());

        thrust::device_vector<unsigned char> d_seen(n);
        thrust::device_vector<int> d_err(1), d_bad_i(1), d_bad_val(1);

        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_seen.data()), 0, n * sizeof(unsigned char)));
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_err.data()), 0, sizeof(int)));

        k_validate_sa<<<1, 1>>>(sa_dev, n, thrust::raw_pointer_cast(d_seen.data()),
                                thrust::raw_pointer_cast(d_err.data()), thrust::raw_pointer_cast(d_bad_i.data()),
                                thrust::raw_pointer_cast(d_bad_val.data()));
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_err = 0;
        CUDA_CHECK(cudaMemcpy(&h_err, thrust::raw_pointer_cast(d_err.data()), sizeof(int), cudaMemcpyDeviceToHost));
        if (h_err != 0)
            throw std::runtime_error("Invalid SA after CUDA-MS.");

        h_sa.resize(n);
        CUDA_CHECK(cudaMemcpy(h_sa.data(), sa_dev, n * sizeof(int), cudaMemcpyDeviceToHost));

        auto t_d2h1 = std::chrono::high_resolution_clock::now();
        m.h_d2h_s = std::chrono::duration<double>(t_d2h1 - t_d2h0).count();

        // Cleanup streams
        for (int s = 0; s < streams; ++s)
            CUDA_CHECK(cudaStreamDestroy(st[s]));
    }

} // namespace cuda_sa_ms
