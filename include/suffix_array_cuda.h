// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>
#include <vector>

/**
 * @brief Computes the suffix array of the input text using CUDA
 * * This function allocates memory on the device, transfers the input text,
 * executes GPU kernels for suffix array construction, and transfers the result back.
 *
 * @param h_text   Host input buffer (text of n bytes)
 * @param h_sa     Output buffer (resized to n), suffix indices sorted lexicographically
 */
void build_suffix_array_cuda(const std::vector<uint8_t> &h_text, std::vector<int> &h_sa);

/**
 * @brief Computes the LCP array on GPU and returns the LRS
 *
 * Assumes that the suffix array `h_sa` has already been computed.
 *
 * @param h_text   Host input buffer (text of n bytes)
 * @param h_sa     Host suffix array (n indices)
 * @param lrs_pos  Output: starting index of the longest repeated substring
 * @param lrs_len  Output: length of the longest repeated substring
 */
void find_lrs_cuda(const std::vector<uint8_t> &h_text, const std::vector<int> &h_sa, int &lrs_pos, int &lrs_len);

/**
 * @brief Frees any persistent device allocations (if used)
 */
void cuda_suffix_array_cleanup();
