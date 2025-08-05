// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

/**
 * Builds the suffix array for a given text using the Manber–Myers algorithm entirely on GPU.
 * Uses async memory allocations and a unified memory pool.
 *
 * @param h_text Input text buffer (host)
 * @param h_sa   Output suffix array (host)
 * @param pool   CUDA memory pool to use for allocations
 */
void build_suffix_array_cuda(const std::vector<uint8_t> &h_text, std::vector<int> &h_sa, cudaMemPool_t pool);

/**
 * Computes the Longest Repeated Substring (LRS) using the suffix array + LCP array entirely on GPU.
 *
 * @param h_text   Input text (host)
 * @param h_sa     Precomputed suffix array (host)
 * @param lrs_pos  Output: starting position of the longest repeated substring
 * @param lrs_len  Output: length of the longest repeated substring
 * @param pool     CUDA memory pool
 */
void find_lrs_cuda(const std::vector<uint8_t> &h_text, const std::vector<int> &h_sa, int &lrs_pos, int &lrs_len,
                   cudaMemPool_t pool);

/**
 * @brief Resets CUDA device state
 */
void cuda_suffix_array_cleanup();
