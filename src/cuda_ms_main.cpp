// Created by Emanuele (https://github.com/Kirito-Emo)

#include "suffix_array.h"
#include "suffix_array_cuda_ms.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Read binary file into vector<uint8_t>
static std::vector<uint8_t> read_binary(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);

    if (!f)
        throw std::runtime_error("Cannot open file: " + path);

    f.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(size);

    if (size && !f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size)))
        throw std::runtime_error("Failed to read file: " + path);

    return buf;
}

static bool is_number_arg(const std::string &s)
{
    if (s.empty())
        return false;

    for (unsigned char c: s)
        if (!std::isdigit(c))
            return false;

    return true;
}

static void print_lrs_hex_preview(const std::vector<uint8_t> &text, int pos, int len, int max_bytes = 16)
{
    if (pos < 0 || len <= 0 || pos >= static_cast<int>(text.size()))
    {
        std::cout << "LRS (hex): <none>\n";
        return;
    }

    const int show = std::min({len, max_bytes, static_cast<int>(text.size()) - pos});
    std::cout << "LRS (hex first " << show << "): ";

    for (int i = 0; i < show; ++i)
        std::cout << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(text[pos + i])
                  << " ";
    std::cout << std::dec << "\n";
}

static int find_lrs_plateau_minpos(const std::vector<int> &sa, const std::vector<int> &lcp, int &best_len, int &pos2)
{
    const int n = (int) sa.size();
    best_len = 0;
    pos2 = -1;
    if (n <= 1)
        return -1;

    int best_i = -1;
    for (int i = 1; i < n; ++i)
    {
        if (lcp[i] > best_len)
        {
            best_len = lcp[i];
            best_i = i;
        }
    }

    if (best_len == 0)
        return -1;

    int L = best_i, R = best_i;

    while (L - 1 >= 1 && lcp[L - 1] == best_len)
        --L;

    while (R + 1 < n && lcp[R + 1] == best_len)
        ++R;

    int best_pos = sa[L - 1];
    for (int j = L; j <= R; ++j)
        best_pos = std::min(best_pos, sa[j]);

    pos2 = -1;
    for (int j = L - 1; j <= R; ++j)
    {
        if (sa[j] != best_pos)
        {
            pos2 = sa[j];
            break;
        }
    }

    return best_pos;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: hpc_cuda_ms <MB_size|file_path> [--streams N]\n";
        return 1;
    }

    // Parse args
    std::string arg = argv[1];
    int streams = 4;
    for (int i = 2; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--streams" && i + 1 < argc)
            streams = std::max(1, std::atoi(argv[++i]));
    }

    std::string path = is_number_arg(arg) ? ("../random_strings/string_" + arg + "MB.bin") : arg;

    std::vector<uint8_t> text;
    try
    {
        text = read_binary(path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading input: " << e.what() << "\n";
        return 1;
    }

    const int n = (int) text.size();
    if (n <= 0)
    {
        std::cerr << "Empty input.\n";
        return 1;
    }

    std::cout << "[CUDA-MS] Input file: " << path << " | size: " << n << " bytes | streams=" << streams << "\n";

    // SA on GPU (multi-stream)
    std::vector<int> sa;
    auto t0 = std::chrono::high_resolution_clock::now();

    try
    {
        cuda_sa_ms::build_suffix_array_cuda_ms(text.data(), n, sa, streams);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in CUDA-MS SA: " << e.what() << "\n";
        return 2;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[CUDA-MS] SA built in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // LCP on CPU (reuse your function)
    std::vector<int> rank(n), lcp(n);
    t0 = std::chrono::high_resolution_clock::now();
    build_lcp(text, sa, rank, lcp);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[Kasai-CPU] LCP built in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    int best_len = 0, pos2 = -1;
    int best_pos = find_lrs_plateau_minpos(sa, lcp, best_len, pos2);
    std::cout << "[Result] Longest repeated substring length = " << best_len << " at position " << best_pos;
    if (pos2 >= 0)
        std::cout << " (another occurrence at " << pos2 << ")";
    std::cout << "\n";
    print_lrs_hex_preview(text, best_pos, best_len);
    if (pos2 >= 0)
        print_lrs_hex_preview(text, pos2, best_len);

    return 0;
}
