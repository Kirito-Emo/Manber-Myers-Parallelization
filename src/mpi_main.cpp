// Created by Emanuele (https://github.com/Kirito-Emo)

#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>
#include "suffix_array.h"
#include "suffix_array_mpi.h"

// Baseline CSV: perf-stats/seq_measurements/seq_summary.csv
// Format: "size,avg_time_compute_pure_s" with size like "100MB"
static double load_seq_baseline_for_mb(int mb, int rank)
{
    const char *env = std::getenv("SEQ_BASELINE_CSV");
    std::string path = env ? std::string(env) : "../perf-stats/seq_measurements/seq_summary.csv";

    if (rank != 0)
        return -1.0; // only rank 0 reads from disk

    std::ifstream fin(path);
    if (!fin)
        return -1.0;

    std::string line;

    if (!std::getline(fin, line))
        return -1.0; // header

    const std::string key = std::to_string(mb) + "MB";
    while (std::getline(fin, line))
    {
        std::stringstream ss(line);
        std::string size, avg;
        if (!std::getline(ss, size, ','))
            continue;

        if (!std::getline(ss, avg, ','))
            continue;

        if (size == key)
        {
            try
            {
                return std::stod(avg);
            }
            catch (...)
            {
                return -1.0;
            }
        }
    }

    return -1.0;
}

static inline double mb_from_bytes(size_t n) { return static_cast<double>(n) / (1024.0 * 1024.0); }

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Rank 0 parses the MB argument and validates it
    int mb = 0;
    if (rank == 0)
    {
        if (argc != 2)
        {
            std::cerr << "Usage: " << argv[0] << " <MB>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Parse numeric argument safely
        auto [ptr, ec] = std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), mb);
        if (ec != std::errc{} || *ptr != '\0' || !(mb == 1 || mb == 50 || mb == 100 || mb == 200 || mb == 500))
        {
            std::cerr << "Error: size must be one of 1, 50, 100, 200 or 500 MB.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast mb to all ranks
    MPI_Bcast(&mb, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute total bytes
    const size_t n = static_cast<size_t>(mb) * 1024 * 1024;
    const int chunk_size = static_cast<int>((n + size - 1) / size);
    const size_t start = static_cast<size_t>(rank) * chunk_size;
    const size_t actual_chunk = (start < n) ? std::min(static_cast<size_t>(chunk_size), n - start) : 0;

    // I/O time
    const double t_io_start = MPI_Wtime();
    std::vector<uint8_t> chunk(actual_chunk);
    // Read the chunk of data from the file
    std::string filename = "../random_strings/string_" + std::to_string(mb) + "MB.bin";
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    if (actual_chunk > 0)
        MPI_File_read_at_all(file, static_cast<MPI_Offset>(start), chunk.data(), static_cast<int>(actual_chunk),
                             MPI_UINT8_T, MPI_STATUS_IGNORE);
    MPI_File_close(&file);
    const double t_io_end = MPI_Wtime();
    const double time_io = t_io_end - t_io_start;

    // Measure host memory allocation overhead
    const double t_alloc_start = MPI_Wtime();
    std::vector<int> sa_local; // Allocate workspace for SA + LCP
    const double t_alloc_end = MPI_Wtime();
    const double time_alloc = t_alloc_end - t_alloc_start;

    // Compute SA (pure) time
    // Build suffix array and LCP array
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_sa_start = MPI_Wtime();
    build_suffix_array_subset(chunk, sa_local);
    const double t_sa_end = MPI_Wtime();
    const double time_sa_local = t_sa_end - t_sa_start;

    // Gather all local SAs to rank 0
    const int local_n = static_cast<int>(sa_local.size());
    std::vector<int> counts(size, 0);
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displs(size, 0);
    int total_sa = 0;
    if (rank == 0)
    {
        for (int r = 0; r < size; ++r)
        {
            displs[r] = total_sa;
            total_sa += counts[r];
        }
    }
    std::vector<int> all_sa(total_sa);

    // Communication
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_comm_start = MPI_Wtime();
    MPI_Gatherv(sa_local.data(), local_n, MPI_INT, (rank == 0 ? all_sa.data() : nullptr),
                (rank == 0 ? counts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_INT, 0,
                MPI_COMM_WORLD);
    const double t_comm_end = MPI_Wtime();
    const double time_comm_local = t_comm_end - t_comm_start;

    // Gather and print times per rank (rank 0 only)
    std::vector<double> times_sa(size), times_comm(size), times_io(size), times_alloc(size);
    MPI_Gather(&time_sa_local, 1, MPI_DOUBLE, times_sa.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&time_comm_local, 1, MPI_DOUBLE, times_comm.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&time_io, 1, MPI_DOUBLE, times_io.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&time_alloc, 1, MPI_DOUBLE, times_alloc.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0: MERGE + LCP + LRS
    double time_merge = 0.0, time_lcp = 0.0, time_lrs_scan = 0.0;
    int max_lcp = 0, pos = 0;

    if (rank == 0)
    {
        // Per-rank printable lines
        for (int r = 0; r < size; ++r)
        {
            std::cout << "rank=" << r << " time_sa_local=" << std::fixed << std::setprecision(6) << times_sa[r]
                      << " time_comm=" << times_comm[r] << " time_io=" << times_io[r]
                      << " time_alloc=" << times_alloc[r] << "\n";
        }

        // Re-load full text (I/O rank-0 only, excluded from compute)
        std::vector<uint8_t> full_text(n);
        {
            std::ifstream fin(filename, std::ios::binary);
            fin.read(reinterpret_cast<char *>(full_text.data()), n);
        }

        // Merge
        const double t_merge_start = MPI_Wtime();
        std::vector<int> sa_global;
        merge_k_sorted_lists(full_text, all_sa, counts, sa_global);
        const double t_merge_end = MPI_Wtime();
        time_merge = t_merge_end - t_merge_start;

        // LCP
        const double t_lcp_start = MPI_Wtime();
        std::vector<int> rk(n), lcp(n);
        build_lcp(full_text, sa_global, rk, lcp);
        const double t_lcp_end = MPI_Wtime();
        time_lcp = t_lcp_end - t_lcp_start;

        // LRS scan
        const double t_lrs_start = MPI_Wtime();
        for (int i = 1; i < static_cast<int>(n); ++i)
        {
            if (lcp[i] > max_lcp)
            {
                max_lcp = lcp[i];
                pos = sa_global[i];
            }
        }
        const double t_lrs_end = MPI_Wtime();
        time_lrs_scan = t_lrs_end - t_lrs_start;

        // Aggregate metrics
        // max(time_sa_local) is the parallel compute bottleneck for the distributed stage
        double time_sa_max = 0.0, time_alloc_max = 0.0, time_io_max = 0.0, time_comm_max = 0.0;
        for (int r = 0; r < size; ++r)
        {
            if (times_sa[r] > time_sa_max)
                time_sa_max = times_sa[r];
            if (times_alloc[r] > time_alloc_max)
                time_alloc_max = times_alloc[r];
            if (times_io[r] > time_io_max)
                time_io_max = times_io[r];
            if (times_comm[r] > time_comm_max)
                time_comm_max = times_comm[r];
        }

        const double time_compute_pure = time_sa_max + time_merge + time_lcp; // no I/O, no comm
        const double time_total_compute = time_alloc_max + time_compute_pure + time_lrs_scan; // no I/O
        const double throughput = mb_from_bytes(n) / (time_compute_pure > 0 ? time_compute_pure : 1.0);
        const double time_transfers_comm = time_comm_max;
        const double memory_overhead_ratio = time_alloc_max / (time_alloc_max + time_compute_pure);

        const double T_seq = load_seq_baseline_for_mb(mb, rank);
        const bool have_baseline = (T_seq > 0.0);
        const double speedup = (have_baseline && time_compute_pure > 0) ? (T_seq / time_compute_pure) : 0.0;
        const double efficiency = (have_baseline && size > 0) ? (speedup / static_cast<double>(size)) : 0.0;

        // Report
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "=== Suffix-Array + LCP (MPI) ===\n";
        std::cout << "size=" << mb << " MB   ranks=" << size << "\n";
        std::cout << "time_io_max=" << time_io_max << " s\n";
        std::cout << "time_alloc_max=" << time_alloc_max << " s\n";
        std::cout << "time_sa_max=" << time_sa_max << " s\n";
        std::cout << "time_merge=" << time_merge << " s\n";
        std::cout << "time_lcp=" << time_lcp << " s\n";
        std::cout << "time_lrs_scan=" << time_lrs_scan << " s\n";
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

        // LRS HEX
        std::cout << "max_lrs_len=" << max_lcp << " pos=" << pos << "\n";
        if (max_lcp > 0)
        {
            std::cout << "LRS (hex): ";
            for (int i = 0; i < max_lcp; ++i)
                std::cout << std::hex << std::uppercase << static_cast<int>(full_text[pos + i]) << ' ';
            std::cout << std::dec << "\n";
        }
        else
            std::cout << "LRS (hex): (none)\n";
    }

    MPI_Finalize();
    return 0;
}
