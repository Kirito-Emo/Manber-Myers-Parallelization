// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cstdint>
#include <vector>

/**
 * Builds the suffix array for a subset of suffixes whose starting positions
 * are given in `starts`.  This function uses the doubling algorithm but only
 * sorts the suffixes in `starts`, comparing full-text ranks.
 *
 * @param text    The input text buffer of length n
 * @param starts  The list of suffix starting indices to be sorted
 * @param sa_out  Output vector, will contain the sorted suffix indices
 *                (same size as `starts`)
 */
void build_suffix_array_subset(const std::vector<uint8_t> &text, const std::vector<int> &starts,
                               std::vector<int> &sa_out);

/**
 * Merges P sorted lists of suffix indices into one global suffix array.
 * Each list is of length counts[r], concatenated in order into `all_sa`.
 * Lexicographic order is determined by comparing suffixes in `text`.
 *
 * @param text     The input text buffer of length n
 * @param all_sa   Concatenation of P sorted SA blocks: block r starts at
 *                 offset displs[r] and has length counts[r]
 * @param counts   The number of entries in each block (size P)
 * @param sa_out   Output vector of length sum(counts), the fully merged SA
 */
void merge_k_sorted_lists(const std::vector<uint8_t> &text, const std::vector<int> &all_sa,
                          const std::vector<int> &counts, std::vector<int> &sa_out);
