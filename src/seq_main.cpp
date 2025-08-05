// Created by Emanuele (https://github.com/Kirito-Emo)

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
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
    if (mb != 1 && mb != 50 && mb != 100 && mb != 200 && mb != 500)
    {
        std::cerr << "Error: size must be one of 1, 50, 100, 200 or 500 MB.\n";
        return 1;
    }

    // Construct filename based on MB size
    std::string filename = "../random_strings/string_" + std::to_string(mb) + "MB.bin";
    const size_t n = static_cast<size_t>(mb) * 1024 * 1024;

    std::vector<uint8_t> text(n);
    std::ifstream fin(filename, std::ios::binary);

    if (!fin)
    {
        std::cerr << "Error: could not open file " << filename << "\n";
        return 1;
    }

    fin.read(reinterpret_cast<char *>(text.data()), n);
    if (static_cast<size_t>(fin.gcount()) != n)
    {
        std::cerr << "Error: file size mismatch. Expected " << n << " bytes.\n";
        return 1;
    }

    fin.close();

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
    if (max_lcp > 0)
    {
        std::cout << "LRS (hex): ";
        for (int i = 0; i < max_lcp; ++i)
            std::cout << std::hex << std::uppercase << static_cast<int>(text[pos_sa + i]) << ' ';
        std::cout << std::dec << "\n";
    }
    else
        std::cout << "LRS: (no repeated substring found)\n";

    return 0;
}
