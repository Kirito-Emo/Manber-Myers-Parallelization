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

    // Allocate all necessary work arrays
    std::vector<int> sa(n), rank(n), cnt(n), next(n), lcp(n);
    std::vector<bool> bh(n), b2h(n);

    // Build suffix array and LCP array
    auto t0 = std::chrono::steady_clock::now(); // Start timing

    build_suffix_array(text, sa, rank, cnt, next, bh, b2h);
    build_lcp(text, sa, rank, lcp);

    // Measuring the time taken to build the suffix and LCP arrays
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    // Report the time taken
    std::cout << "Time taken to build suffix array and LCP array\n";
    std::cout << "n=" << n << "   time_build=" << elapsed_s << " s\n\n";

    // Find the maximum LCP value—and its position
    int max_lcp = 0, idx = 0;
    for (int i = 1; i < n; ++i)
    {
        if (lcp[i] > max_lcp)
        {
            max_lcp = lcp[i];
            idx = sa[i];
        }
    }

    // Output the result
    std::cout << "LCP array report\n";
    std::cout << "n=" << n << "   time_build=" << elapsed_s << " s"
              << "   max_lcp=" << max_lcp << "   pos=" << idx << "\n";

    return 0;
}
