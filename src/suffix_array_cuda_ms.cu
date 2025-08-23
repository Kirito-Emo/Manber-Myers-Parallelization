// Created by Emanuele (https://github.com/Kirito-Emo)

#include "suffix_array_cuda_ms.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <vector>

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

namespace cuda_sa_ms
{

// Convert byte flag -> int (needed by exclusive_scan)
struct UCharToInt
{
    __host__ __device__
    int operator()(unsigned char x) const
    {
        return x ? 1 : 0;
    }
};

// Comparator on SA indices using (rank[i], rank[i+k])
struct SAKeyLess
{
    const int* rank;   // device pointer
    int        n;
    int        k;

    __host__ __device__
    bool operator()(int a, int b) const
    {
        int r1a = rank[a];
        int r1b = rank[b];
        if (r1a != r1b) return r1a < r1b;

        int r2a = (a + k < n) ? (rank[a + k] + 1) : 0;
        int r2b = (b + k < n) ? (rank[b + k] + 1) : 0;
        return r2a < r2b;
    }
};

// ---- Kernels ----------------------------------------------------------------

// Initialize ranks from text bytes
__global__ void k_init_ranks(const uint8_t* text, int* rank, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        rank[i] = static_cast<int>(text[i]);
}

// Build 32-bit secondary keys on subrange: key = rank[i+k] + 1 (0 if out-of-bounds)
__global__ void k_build_key_r2_range(const int* rank, uint32_t* key, int n, int k, int start, int len)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < len)
    {
        int i = start + t;
        key[i] = (i + k < n) ? static_cast<uint32_t>(rank[i + k] + 1) : 0u;
    }
}

// Mark group boundaries by comparing (r1,r2) via ranks (1-byte flags)
__global__ void k_mark_groups_by_rank_u8(const int* sa, const int* rank, int n, int k, unsigned char* flags)
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

        flags[i] = (r1a != r1b) || (r2a != r2b);
    }
}

// Scatter group IDs (in sorted order) back to text index order
__global__ void k_scatter_ranks(const int* sa_sorted, const int* rank_sorted, int* new_rank, int n)
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
__global__ void k_validate_sa(const int* sa, int n, unsigned char* seen, int* err, int* bad_i, int* bad_val)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            int v = sa[i];

            if (v < 0 || v >= n)
            {
                *err = 101; *bad_i = i; *bad_val = v;
                return;
            }

            if (seen[v])
            {
                *err = 102; *bad_i = i; *bad_val = v;
                return;
            }

            seen[v] = 1;
        }

        *err = 0;
    }
}

// ---- Helpers ----------------------------------------------------------------

static inline void compute_chunks(int n, int S, std::vector<int>& offs, std::vector<int>& lens)
{
    offs.resize(S);
    lens.resize(S);
    int base = n / S;
    int rem  = n % S;
    int acc  = 0;

    for (int s = 0; s < S; ++s)
    {
        int len = base + (s < rem ? 1 : 0);
        offs[s] = acc;
        lens[s] = len;
        acc += len;
    }
}

// ---- Public API --------------------------------------------------------------

void build_suffix_array_cuda_ms(const uint8_t* h_text, int n, std::vector<int>& h_sa, int streams)
{
    if (n <= 0)
    {
        h_sa.clear();
        return;
    }

    if (streams <= 0)
        streams = 1;

    // Device buffers (slim)
    thrust::device_vector<uint8_t>       d_text(h_text, h_text + n);
    thrust::device_vector<int>           d_rank(n), d_new_rank(n);
    thrust::device_vector<int>           d_rank_scan(n);
    thrust::device_vector<unsigned char> d_flags(n);      // 1-byte flags
    thrust::device_vector<uint32_t>      d_key32(n);      // reused per pass
    thrust::device_vector<int>           d_sa_A(n), d_sa_B(n); // ping-pong SA

    const int BLOCK = 256;
    int gridN = (n + BLOCK - 1) / BLOCK;

    // Init ranks from text
    k_init_ranks<<<gridN, BLOCK>>>(thrust::raw_pointer_cast(d_text.data()), thrust::raw_pointer_cast(d_rank.data()), n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create streams
    std::vector<cudaStream_t> st(streams);
    for (int s = 0; s < streams; ++s)
        CUDA_CHECK(cudaStreamCreateWithFlags(&st[s], cudaStreamNonBlocking));

    // Chunk layout
    std::vector<int> offs, lens;
    compute_chunks(n, streams, offs, lens);

    // Limit parallel merges to cap scratch usage (2 is a good default)
    const int MAX_PARALLEL_MERGES = std::min(streams, 2);

    // Doubling loop
    for (int k = 1; ; k <<= 1)
    {
        // Seed SA_A with [0..n-1]
        thrust::sequence(d_sa_A.begin(), d_sa_A.end());

        // ---- Per-chunk two-pass stable sorting (r2 then r1) -----------------
        for (int s = 0; s < streams; ++s)
        {
            int off  = offs[s];
            int len  = lens[s];
            int grid = (len + BLOCK - 1) / BLOCK;

            // Build r2 keys on subrange
            k_build_key_r2_range<<<grid, BLOCK, 0, st[s]>>>(
                thrust::raw_pointer_cast(d_rank.data()),
                thrust::raw_pointer_cast(d_key32.data()),
                n, k, off, len);
        }

        for (int s = 0; s < streams; ++s)
            CUDA_CHECK(cudaStreamSynchronize(st[s]));

        // Stable sort by r2: (key32, sa_A) in each chunk
        for (int s = 0; s < streams; ++s)
        {
            int off = offs[s], len = lens[s];
            auto pol = thrust::cuda::par.on(st[s]);
            thrust::stable_sort_by_key(pol,
                d_key32.begin() + off, d_key32.begin() + off + len,
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
            thrust::gather(pol,
                d_sa_A.begin() + off, d_sa_A.begin() + off + len,
                d_rank.begin(),
                d_key32.begin() + off);

            thrust::stable_sort_by_key(pol,
                d_key32.begin() + off, d_key32.begin() + off + len,
                d_sa_A.begin() + off);
        }
        for (int s = 0; s < streams; ++s) CUDA_CHECK(cudaStreamSynchronize(st[s]));

        // ---- Pairwise merges of SA runs using comparator on ranks ----------
        bool in_A = true;    // current runs live in SA_A
        int  runs = streams;

        std::vector<int> cur_offs = offs;
        std::vector<int> cur_lens = lens;

        while (runs > 1)
        {
            int pairs = runs / 2;

            // Launch merges in waves to limit parallel scratch usage
            int launched = 0;
            for (int p = 0; p < pairs; ++p)
            {
                int sidx = launched % MAX_PARALLEL_MERGES;
                auto pol = thrust::cuda::par.on(st[sidx]);

                int a = 2 * p;
                int b = a + 1;

                int offA = cur_offs[a];
                int lenA = cur_lens[a];
                int offB = cur_offs[b];
                int lenB = cur_lens[b];

                auto saIn  = in_A ? d_sa_A.begin() : d_sa_B.begin();
                auto saOut = in_A ? d_sa_B.begin() : d_sa_A.begin();

                // Comparator uses device pointer to rank and current k
                SAKeyLess cmp{ thrust::raw_pointer_cast(d_rank.data()), n, k };

                // Merge two runs into output at offA
                thrust::merge(pol,
                    saIn + offA, saIn + offA + lenA,
                    saIn + offB, saIn + offB + lenB,
                    saOut + offA, cmp);

                ++launched;

                // synchronize wave
                if (launched == MAX_PARALLEL_MERGES || p == pairs - 1)
                {
                    for (int i = 0; i < launched; ++i)
                        CUDA_CHECK(cudaStreamSynchronize(st[i]));
                    launched = 0;
                }
            }

            // Odd run: copy-through into buffer that will be current on next level
            if (runs % 2 == 1)
            {
                int a = runs - 1;
                int offA = cur_offs[a];
                int lenA = cur_lens[a];

                if (in_A)
                {
                    CUDA_CHECK(cudaMemcpyAsync(
                        thrust::raw_pointer_cast(d_sa_B.data()) + offA,
                        thrust::raw_pointer_cast(d_sa_A.data()) + offA,
                        sizeof(int) * lenA, cudaMemcpyDeviceToDevice, st[0]));
                }
                else
                {
                    CUDA_CHECK(cudaMemcpyAsync(
                        thrust::raw_pointer_cast(d_sa_A.data()) + offA,
                        thrust::raw_pointer_cast(d_sa_B.data()) + offA,
                        sizeof(int) * lenA, cudaMemcpyDeviceToDevice, st[0]));
                }

                CUDA_CHECK(cudaStreamSynchronize(st[0]));
            }

            // Build next level run list
            std::vector<int> next_offs;
            std::vector<int> next_lens;
            next_offs.reserve((runs + 1) / 2);
            next_lens.reserve((runs + 1) / 2);

            for (int p = 0; p < pairs; ++p)
            {
                int a = 2 * p;
                int b = a + 1;
                next_offs.push_back(cur_offs[a]);
                next_lens.push_back(cur_lens[a] + cur_lens[b]);
            }

            if (runs % 2 == 1)
            {
                int a = runs - 1;
                next_offs.push_back(cur_offs[a]);
                next_lens.push_back(cur_lens[a]);
            }

            cur_offs.swap(next_offs);
            cur_lens.swap(next_lens);
            runs = static_cast<int>(cur_offs.size());
            in_A = !in_A; // outputs written to the opposite buffer
        }

        // Pick current SA buffer after merges
        const int* sa_sorted = in_A
            ? thrust::raw_pointer_cast(d_sa_A.data())
            : thrust::raw_pointer_cast(d_sa_B.data());

        // ---- Mark flags, scan, scatter to new ranks ------------------------
        k_mark_groups_by_rank_u8<<<gridN, BLOCK>>>(
            sa_sorted,
            thrust::raw_pointer_cast(d_rank.data()),
            n, k,
            thrust::raw_pointer_cast(d_flags.data())
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::transform(d_flags.begin(), d_flags.end(), d_rank_scan.begin(), UCharToInt{});
        thrust::exclusive_scan(d_rank_scan.begin(), d_rank_scan.end(), d_rank_scan.begin());

        k_scatter_ranks<<<gridN, BLOCK>>>(
            sa_sorted,
            thrust::raw_pointer_cast(d_rank_scan.data()),
            thrust::raw_pointer_cast(d_new_rank.data()), n
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        int max_rank = thrust::reduce(d_new_rank.begin(), d_new_rank.end(), -1, thrust::maximum<int>());

        d_rank.swap(d_new_rank);
        if (max_rank == n - 1)
            break;

        if (k > n)
            break;
    }

    // Validate and copy back SA (we expect it in SA_A if in_A==true, else SA_B)
    {
        const int* sa_dev = thrust::raw_pointer_cast(d_sa_A.data());

        thrust::device_vector<unsigned char> d_seen(n);
        thrust::device_vector<int> d_err(1), d_bad_i(1), d_bad_val(1);

        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_seen.data()), 0, n * sizeof(unsigned char)));
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_err.data()), 0, sizeof(int)));

        k_validate_sa<<<1,1>>>(sa_dev, n,
                               thrust::raw_pointer_cast(d_seen.data()),
                               thrust::raw_pointer_cast(d_err.data()),
                               thrust::raw_pointer_cast(d_bad_i.data()),
                               thrust::raw_pointer_cast(d_bad_val.data()));
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_err = 0;
        CUDA_CHECK(cudaMemcpy(&h_err, thrust::raw_pointer_cast(d_err.data()), sizeof(int), cudaMemcpyDeviceToHost));
        if (h_err != 0)
        {
            // Try alternate buffer before failing
            sa_dev = thrust::raw_pointer_cast(d_sa_B.data());
            CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_seen.data()), 0, n * sizeof(unsigned char)));
            CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(d_err.data()), 0, sizeof(int)));
            k_validate_sa<<<1,1>>>(sa_dev, n,
                                   thrust::raw_pointer_cast(d_seen.data()),
                                   thrust::raw_pointer_cast(d_err.data()),
                                   thrust::raw_pointer_cast(d_bad_i.data()),
                                   thrust::raw_pointer_cast(d_bad_val.data()));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_err, thrust::raw_pointer_cast(d_err.data()), sizeof(int), cudaMemcpyDeviceToHost));
            if (h_err != 0)
                throw std::runtime_error("Invalid SA after CUDA-MS (slim).");
        }

        h_sa.resize(n);
        CUDA_CHECK(cudaMemcpy(h_sa.data(), sa_dev, n * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Cleanup streams
    for (int s = 0; s < streams; ++s)
        CUDA_CHECK(cudaStreamDestroy(st[s]));
}

} // namespace cuda_sa_ms
