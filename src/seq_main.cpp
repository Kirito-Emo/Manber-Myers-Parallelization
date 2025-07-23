// Created by [Emanuele](https://github.com/Kirito-Emo)

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include "seq_suffix.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input-file>\n";
        return 1;
    }

    // Read the input file
    std::ifstream ifs(argv[1], std::ios::binary);
    std::string input((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    int n = input.size();

    std::vector<int> text(n);
    for (int i = 0; i < n; ++i)
        text[i] = static_cast<unsigned char>(input[i]);

    // --- SUFFIX ARRAY ---

    // Allocate workspace
    std::vector<int> sa(n), rank(n), cnt(n), next(n), lcp(n);
    std::vector<bool> bh(n), b2h(n);

    // Build suffix array and LCP array
    auto t0 = std::chrono::steady_clock::now(); // Start timing

    build_suffix_array(text, sa, rank, cnt, next, bh, b2h);
    build_lcp(text, sa, rank, lcp);

    // Measuring the time taken to build the suffix and LCP arrays
    auto t1 = std::chrono::steady_clock::now();
    double time_sa = std::chrono::duration<double>(t1 - t0).count();

    // Find max LCP and its position
    int max_lcp = 0, pos_sa = 0;
    for (int i = 1; i < n; ++i)
    {
        if (lcp[i] > max_lcp)
        {
            max_lcp = lcp[i];
            pos_sa = sa[i];
        }
    }

    // Report Suffix-Array + LCP results
    std::cout << "=== Suffix‐Array + LCP ===\n";
    std::cout << "n=" << n << " MB   time_build=" << time_sa << " s\n";
    std::cout << "max_lrs_len=" << max_lcp << "   pos=" << pos_sa << "\n\n";

    // --- SUFFIX TREE ---
    auto t2 = std::chrono::steady_clock::now();
    SuffixTree st(text); // Build the suffix tree
    auto [len_st, pos_st] = st.LRS(); // Find the longest repeated substring
    auto t3 = std::chrono::steady_clock::now();

    double time_st = std::chrono::duration<double>(t3 - t2).count();

    // Report Suffix-Tree + LRS results
    std::cout << "=== Suffix‐Tree + LRS ===\n";
    std::cout << "n=" << n << " MB   time_build=" << time_st << " s\n";
    std::cout << "max_lrs_len=" << len_st << "   pos=" << pos_st << "\n";

    return 0;
}
