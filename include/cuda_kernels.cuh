// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>

// Default CUDA block size (shared by all kernels)
#define BLOCK_SIZE 256

// CUDA error checking utility (declared in cuda_utils.h)
#include "cuda_utils.h"

// ==== KERNEL DECLARATIONS ====

// Initialize ranks from text[i] (rank[i] = text[i])
void init_ranks_from_text(const uint8_t *d_text, int *d_rank, int n);

// Compute key[i] = pack(rank[i], rank[i + k]) to sort suffixes
void compute_rank_pairs(const int *d_rank, int *d_keys, int n, int k);

// Compute new ranks after sorting: requires prefix-sum logic
void update_ranks(const int *d_sa, const int *d_rank, int *d_new_rank, int n);

// Compute rank[i] = inverse of sa[i]
__global__ void compute_rank_kernel(const int *sa, int *rank, int n);

// Compute LCP array using Kasai’s algorithm (GPU variant)
__global__ void compute_lcp_kernel(const uint8_t *text, const int *sa, const int *rank, int *lcp, int n);
