// Created by [Emanuele](https://github.com/Kirito-Emo)

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "suffix_array.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <size-in-MB>\n";
        return 1;
    }

    // Parse the desired size in MB from command line
    const int mb = std::stoi(argv[1]);
    if (mb != 1 && mb != 5 && mb != 10 && mb != 50 && mb != 100 && mb != 500)
    {
        std::cerr << "Error: size must be one of 1, 5, 10, 50, 100, or 500 MB.\n";
        return 1;
    }

    // Compute total number of bytes = MB * 1024 * 1024
    const size_t n = static_cast<size_t>(mb) * 1024 * 1024;

    // Prepare text buffer in RAM (1 byte per character)
    std::vector<uint8_t> text(n);

    // Use of mt19937_64 (high-quality non-CS PRNG 64bit Mersenne Twister)
    std::mt19937_64 gen(123456789ULL); // reproducible randomness due to fixed seed
    std::uniform_int_distribution<int> dist(0, 255); // uniform [0..255]

    // Fill with pseudorandom bytes
    for (size_t i = 0; i < n; ++i)
        text[i] = static_cast<uint8_t>(dist(gen));

    // Allocate workspace for SA + LCP
    std::vector<int> sa(n), rank(n), cnt(n), next(n), lcp(n);
    std::vector<bool> bh(n), b2h(n);

    // Build suffix array and LCP array
    auto t0 = std::chrono::steady_clock::now(); // Start timing

    build_suffix_array(text, sa, rank, cnt, next, bh, b2h);
    build_lcp(text, sa, rank, lcp);

    // Measuring the time taken to build the suffix and LCP arrays
    auto t1 = std::chrono::steady_clock::now();
    double time_sa = std::chrono::duration<double>(t1 - t0).count();

    // Find Longest Repeated Substring (LRS) via LCP
    int max_lcp = 0, pos_sa = 0;
    for (int i = 1; i < static_cast<int>(n); ++i)
    {
        if (lcp[i] > max_lcp)
        {
            max_lcp = lcp[i];
            pos_sa = sa[i];
        }
    }

    // Report Suffix-Array + LCP results
    std::cout << "=== Suffix‐Array + LCP ===\n";
    std::cout << "size=" << mb << " MB   "
              << "time_build=" << time_sa << " s\n";
    std::cout << "max_lrs_len=" << max_lcp << "   pos=" << pos_sa << "\n";

    return 0;
}
