// Created by Emanuele (https://github.com/Kirito-Emo)

#include "suffix_array.h"
#include "suffix_array_cuda.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

// Determine if argument is an integer number (ASCII digits only)
static bool is_number_arg(const std::string &s)
{
    if (s.empty())
        return false;

    for (unsigned char c: s)
        if (!std::isdigit(c))
            return false;

    return true;
}

static inline double mb_from_bytes(size_t n) { return static_cast<double>(n) / (1024.0 * 1024.0); }

static double load_seq_baseline_for_mb_arg(const std::string &arg)
{
    for (unsigned char c: arg)
        if (!std::isdigit(c))
            return -1.0;

    int mb = std::stoi(arg);
    const char *env = std::getenv("SEQ_BASELINE_CSV");
    std::string path = env ? std::string(env) : "../perf-stats/seq_measurements/seq_summary.csv";
    std::ifstream fin(path);
    if (!fin)
        return -1.0;

    std::string line;
    if (!std::getline(fin, line))
        return -1.0; // skip header

    const std::string key = std::to_string(mb) + "MB";
    while (std::getline(fin, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> cols;

        // Split by comma
        while (std::getline(ss, token, ','))
            cols.push_back(token);

        if (cols.size() > 6 && cols[0] == key)
        {
            try
            {
                return std::stod(cols[6]);
            }
            catch (...)
            {
                return -1.0;
            }
        }
    }

    return -1.0;
}

// Hex preview of LRS
static void print_lrs_hex_preview(const std::vector<uint8_t> &text, int pos, int len)
{
    if (pos < 0 || len <= 0 || pos >= static_cast<int>(text.size()))
    {
        std::cout << "LRS (hex): <none>\n";
        return;
    }
    std::cout << "LRS (hex): ";
    for (int i = 0; i < len; ++i)
        std::cout << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(text[pos + i])
                  << " ";
    std::cout << std::dec << "\n";
}

// Find LRS position using plateau-aware tie-breaking (min position in text)
// Returns best_pos; writes best_len and another occurrence pos2 (or -1 if none)
static int find_lrs_plateau_minpos(const std::vector<int> &sa, const std::vector<int> &lcp, int &best_len, int &pos2)
{
    const int n = static_cast<int>(sa.size());
    best_len = 0;
    pos2 = -1;
    if (n <= 1)
        return -1;

    // Find max LCP and its first index
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

    int L = best_i, R = best_i; // Expand to full plateau [L..R] where lcp == best_len
    while (L - 1 >= 1 && lcp[L - 1] == best_len)
        --L;

    while (R + 1 < n && lcp[R + 1] == best_len)
        ++R;

    // Suffixes involved are sa[L-1..R]
    int best_pos = sa[L - 1];
    for (int j = L; j <= R; ++j)
        if (sa[j] < best_pos)
            best_pos = sa[j];

    // Find another occurrence for sanity print
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
        std::cerr << "Usage: hpc_cuda <MB_size | file_path>\n";
        return 1;
    }

    // Map numeric argument to canonical file path, else treat as direct path
    std::string arg = argv[1];
    std::string path = is_number_arg(arg) ? ("../random_strings/string_" + arg + "MB.bin") : arg;

    // I/O (host)
    auto t_io0 = std::chrono::steady_clock::now();
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
    auto t_io1 = std::chrono::steady_clock::now();
    const double time_io = std::chrono::duration<double>(t_io1 - t_io0).count();

    const int n = static_cast<int>(text.size());
    if (n <= 0)
    {
        std::cerr << "Empty input.\n";
        return 1;
    }

    std::cout << "[CUDA] Input file: " << path << " | size: " << n << " bytes\n";

    // Build suffix array on GPU + metrics
    std::vector<int> sa;
    cuda_sa::Metrics cm{};
    try
    {
        cuda_sa::build_suffix_array_cuda(text.data(), n, sa, cm);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in CUDA SA: " << e.what() << "\n";
        return 2;
    }

    // LCP on CPU (Kasai)
    auto t_lcp0 = std::chrono::steady_clock::now();
    std::vector<int> rank(n), lcp(n);
    try
    {
        build_lcp(text, sa, rank, lcp);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in CPU Kasai: " << e.what() << "\n";
        return 3;
    }
    auto t_lcp1 = std::chrono::steady_clock::now();
    const double time_lcp_cpu = std::chrono::duration<double>(t_lcp1 - t_lcp0).count();
    std::cout << "[Kasai-CPU] LCP built in " << time_lcp_cpu << " s\n";

    // Compute LRS with plateau-aware tie-breaking
    int best_len = 0, pos2 = -1;
    int best_pos = find_lrs_plateau_minpos(sa, lcp, best_len, pos2);

    // Pure computing (no transfers): kernel GPU + Kasai CPU
    const double time_compute_pure = cm.gpu_kernel_s + time_lcp_cpu;

    // Transfers/communications: copies H<->D
    const double time_transfers_comm = cm.h_h2d_s + cm.h_d2h_s;

    // Memory overhead: alloc host + alloc device + copies / (alloc+compute)
    const double time_alloc_total = cm.h_alloc_s; // alloc host+device
    const double memory_overhead_ratio =
            (time_alloc_total + time_transfers_comm) / (time_alloc_total + time_compute_pure);

    // Total computing time (without I/O): alloc + pure computing
    const double time_total_compute = time_alloc_total + time_compute_pure;

    // Pure computing throughput (MB/s)
    const double throughput = mb_from_bytes(static_cast<size_t>(n)) / (time_compute_pure > 0 ? time_compute_pure : 1.0);

    // Sequential baseline
    const double T_seq = load_seq_baseline_for_mb_arg(arg);
    const bool have_baseline = (T_seq > 0.0);
    const double speedup = (have_baseline && time_compute_pure > 0) ? (T_seq / time_compute_pure) : 0.0;
    const double efficiency = speedup; // 1 GPU = 1 "resource"

    // Report
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Suffix-Array + LCP (CUDA single-stream) ===\n";
    std::cout << "time_io=" << time_io << " s\n";
    std::cout << "time_alloc_host_dev=" << time_alloc_total << " s\n";
    std::cout << "time_h2d=" << cm.h_h2d_s << " s\n";
    std::cout << "time_kernel_gpu=" << cm.gpu_kernel_s << " s\n";
    std::cout << "time_lcp_cpu=" << time_lcp_cpu << " s\n";
    std::cout << "time_d2h=" << cm.h_d2h_s << " s\n";
    std::cout << "time_compute_pure=" << time_compute_pure << " s\n";
    std::cout << "time_total_compute=" << time_total_compute << " s\n";
    std::cout << "time_transfers_comm=" << time_transfers_comm << " s\n";
    std::cout << "throughput=" << throughput << " MB/s\n";
    if (have_baseline)
    {
        std::cout << "speedup=" << speedup << "\n";
        std::cout << "efficiency=" << (efficiency * 100.0) << " %\n";
    }
    else
    {
        std::cout << "speedup=n/a (no baseline)\n";
        std::cout << "efficiency=n/a (no baseline)\n";
    }
    std::cout << "memory_overhead_ratio=" << (memory_overhead_ratio * 100.0) << " %\n";

    // LRS hex
    std::cout << "[Result] LRS length=" << best_len << "   pos=" << best_pos;
    if (pos2 >= 0)
        std::cout << " (another occurrence at " << pos2 << ")";
    std::cout << "\n";

    // Hex preview of the first bytes of LRS
    print_lrs_hex_preview(text, best_pos, best_len);

    return 0;
}
