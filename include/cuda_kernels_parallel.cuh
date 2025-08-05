// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Define a tunable block size
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ==== KERNEL DECLARATIONS ====

// Initialize rank[i] = text[i] asynchronously on stream
void init_ranks_from_text(const uint8_t *d_text, int *d_rank, int n, cudaStream_t stream);

// Compute keys[i] = (rank[i] << 16) | rank[i + k] on stream
void compute_rank_pairs(const int *d_rank, int *d_keys, int n, int k, cudaStream_t stream);

// Update ranks based on sorted suffixes (with prefix sum)
void update_ranks(const int *d_sa, const int *d_rank, int *d_new_rank, int n, cudaStream_t stream);

// Compute rank[sa[i]] = i (inverse suffix array)
__global__ void compute_rank_kernel(const int *__restrict__ sa, int *__restrict__ rank, int n);

// Kasai algorithm on GPU to compute LCP
__global__ void compute_lcp_kernel(const uint8_t *__restrict__ text, const int *__restrict__ sa,
                                   const int *__restrict__ rank, int *__restrict__ lcp, int n);
