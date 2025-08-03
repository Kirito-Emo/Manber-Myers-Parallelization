// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>
#include <vector>

/**
 * Builds the suffix array for the local chunk held by the process.
 * @param chunk   The chunk of text (subset of the global input)
 * @param sa_out  Output suffix array (same length as chunk)
 */
void build_suffix_array_subset(const std::vector<uint8_t> &chunk, std::vector<int> &sa_out);

/**
 * Merges sorted local suffix arrays into a global suffix array.
 * Offsets are adjusted based on chunk index.
 * @param text     Full original text (only available on rank 0)
 * @param all_sa   Concatenation of local suffix arrays from all ranks
 * @param counts   Number of suffixes from each process
 * @param sa_out   Merged suffix array
 */
void merge_k_sorted_lists(const std::vector<uint8_t> &text, const std::vector<int> &all_sa,
                          const std::vector<int> &counts, std::vector<int> &sa_out);
