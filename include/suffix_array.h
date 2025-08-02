// Created by [Emanuele](https://github.com/Kirito-Emo)

#ifndef SUFFIX_ARRAY_H
#define SUFFIX_ARRAY_H

#include <cstdint>
#include <vector>

/**
 * @brief Build the suffix array of a text in O(n log n) using Manber & Myers algorithm
 *
 * @param text  Input text as a vector of bytes (each 0–255)
 * @param sa    Output: the suffix array (indices of sorted suffixes)
 * @param rank  Temporary workspace / inverse of sa during construction
 * @param cnt   Workspace for bucket counts
 * @param next  Workspace for bucket boundaries
 * @param bh    Bucket head flags
 * @param b2h   Secondary bucket head flags
 */
void build_suffix_array(const std::vector<uint8_t> &text, std::vector<int> &sa, std::vector<int> &rank,
                        std::vector<int> &cnt, std::vector<int> &next, std::vector<bool> &bh, std::vector<bool> &b2h);

/**
 * @brief Build the LCP (Longest Common Prefix) array in O(n) using Kasai et al.'s algorithm
 *
 * @param text  The same input text as for build_suffix_array
 * @param sa    The suffix array produced by build_suffix_array
 * @param rank  Workspace / inverse of sa
 * @param lcp   Output: lcp[i] = length of LCP(sa[i], sa[i-1])
 */
void build_lcp(const std::vector<uint8_t> &text, const std::vector<int> &sa, std::vector<int> &rank,
               std::vector<int> &lcp);

#endif // SUFFIX_ARRAY_H
