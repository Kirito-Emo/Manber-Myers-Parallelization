// Created by Emanuele (https://github.com/Kirito-Emo)

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "suffix_array.h"

static inline double mb_from_bytes(size_t n) { return static_cast<double>(n) / (1024.0 * 1024.0); }

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <size-in-MB>\n";
        return 1;
    }

    // Parse the desired size in MB from command line
    const int target_mb = std::stoi(argv[1]);
    if (target_mb != 1 && target_mb != 50 && target_mb != 100 && target_mb != 200 && target_mb != 500)
    {
        std::cerr << "Error: size must be one of 1, 50, 100, 200 or 500 MB.\n";
        return 1;
    }

    // Construct filename based on MB size
    std::string filename = "../random_strings/string_" + std::to_string(target_mb) + "MB.bin";

    // Measure file I/O
    auto t_io_start = std::chrono::steady_clock::now();
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    if (!fin)
    {
        std::cerr << "Error: could not open file " << filename << "\n";
        return 1;
    }
    const size_t n = fin.tellg();
    fin.seekg(0, std::ios::beg);

    std::vector<uint8_t> text(n);
    fin.read(reinterpret_cast<char *>(text.data()), n);
    fin.close();
    auto t_io_end = std::chrono::steady_clock::now();
    const double time_io = std::chrono::duration<double>(t_io_end - t_io_start).count();

    // Measure host memory allocation overhead
    auto t_alloc_start = std::chrono::steady_clock::now();
    // Allocate workspace for SA + LCP
    std::vector<int> sa(n), rank(n), cnt(n), next(n), lcp(n);
    std::vector<bool> bh(n), b2h(n);
    auto t_alloc_end = std::chrono::steady_clock::now();
    const double time_alloc = std::chrono::duration<double>(t_alloc_end - t_alloc_start).count();

    // Compute SA (pure) time
    auto t_sa_start = std::chrono::steady_clock::now();
    build_suffix_array(text, sa, rank, cnt, next, bh, b2h);
    auto t_sa_end = std::chrono::steady_clock::now();
    const double time_sa = std::chrono::duration<double>(t_sa_end - t_sa_start).count();

    // Compute LCP (pure) time
    auto t_lcp_start = std::chrono::steady_clock::now();
    build_lcp(text, sa, rank, lcp);
    auto t_lcp_end = std::chrono::steady_clock::now();
    const double time_lcp = std::chrono::duration<double>(t_lcp_end - t_lcp_start).count();

    // LRS
    auto t_lrs_start = std::chrono::steady_clock::now();
    int max_lcp = 0, pos_sa = 0;
    for (size_t i = 1; i < n; ++i)
    {
        if (lcp[i] > max_lcp)
        {
            max_lcp = lcp[i];
            pos_sa = sa[i];
        }
    }
    auto t_lrs_end = std::chrono::steady_clock::now();
    const double time_lrs_scan = std::chrono::duration<double>(t_lrs_end - t_lrs_start).count();

    // Aggregate metrics
    const double time_compute_pure = time_sa + time_lcp; // "pure compute": SA + LCP
    const double time_total_compute = time_alloc + time_compute_pure + time_lrs_scan; // no I/O time
    const double throughput_MBps = mb_from_bytes(n) / time_compute_pure;
    const double time_transfers_comm = 0.0; // No transfers/communications in sequential version

    // Sequential baseline: p = 1 -> speedup = 1, efficiency = 100%
    const double speedup = 1.0;
    const double efficiency = 1.0;

    // Memory overhead relative to compute (host-side allocations only)
    const double memory_overhead_ratio = time_alloc / (time_alloc + time_compute_pure);

    // Report
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Suffix-Array + LCP (Sequential) ===\n";
    std::cout << "target_size=" << target_mb << " MB (string=" << mb_from_bytes(n) << " MB)\n";
    std::cout << "time_io=" << time_io << " s\n";
    std::cout << "time_alloc=" << time_alloc << " s\n";
    std::cout << "time_sa=" << time_sa << " s\n";
    std::cout << "time_lcp=" << time_lcp << " s\n";
    std::cout << "time_lrs_scan=" << time_lrs_scan << " s\n";
    std::cout << "time_compute_pure=" << time_compute_pure << " s\n";
    std::cout << "time_total_compute=" << time_total_compute << " s\n";
    std::cout << "time_transfers_comm=" << time_transfers_comm << " s\n";
    std::cout << "throughput=" << throughput_MBps << " MB/s\n";
    std::cout << "speedup=" << speedup << "\n";
    std::cout << "efficiency=" << (efficiency * 100.0) << " %\n";
    std::cout << "memory_overhead_ratio=" << (memory_overhead_ratio * 100.0) << " %\n";

    // Print LRS info (also in hex)
    std::cout << "max_lrs_len=" << max_lcp << "   pos=" << pos_sa << "\n";
    if (max_lcp > 0)
    {
        std::cout << "LRS (hex): ";
        for (int i = 0; i < max_lcp; ++i)
            std::cout << std::hex << std::uppercase << static_cast<int>(text[pos_sa + i]) << ' ';
        std::cout << std::dec << "\n";
    }
    else
        std::cout << "LRS (hex): (no repeated substring found)\n";

    return 0;
}
