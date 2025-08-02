// Created by Emanuele (https://github.com/Kirito-Emo)

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>
#include "suffix_array.h"
#include "suffix_array_mpi.h"

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
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
        if (ec != std::errc{} || *ptr != '\0')
        {
            std::cerr << "Error: argument must be an integer\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Validate allowed sizes
        if (!(mb == 1 || mb == 5 || mb == 10 || mb == 50 || mb == 100 || mb == 500))
        {
            std::cerr << "Error: size must be one of 1, 5, 10, 50, 100, or 500 MB.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast mb to all ranks
    MPI_Bcast(&mb, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute total bytes
    size_t n = static_cast<size_t>(mb) * 1024 * 1024;

    // Generate pseudo-random text with fixed seed
    std::vector<uint8_t> text(n);
    std::mt19937_64 gen(123456789ULL); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 255);

    for (size_t i = 0; i < n; ++i)
    {
        text[i] = static_cast<uint8_t>(dist(gen));
    }

    // Decompose suffix indices among ranks
    std::vector<int> starts;

    for (size_t i = rank; i < n; i += size)
    {
        starts.push_back(i);
    }

    // Build local suffix array subset
    std::vector<int> sa_local;

    auto t0 = std::chrono::steady_clock::now();

    build_suffix_array_subset(text, starts, sa_local);

    auto t1 = std::chrono::steady_clock::now();
    double time_build = std::chrono::duration<double>(t1 - t0).count();

    // Gather all local SAs to rank 0
    std::vector<int> counts(size), displs(size);
    int local_n = sa_local.size();
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> all_sa;

    if (rank == 0)
    {
        displs[0] = 0;
        for (int r = 1; r < size; ++r)
            displs[r] = displs[r - 1] + counts[r - 1];
        all_sa.resize(n);
    }

    MPI_Gatherv(sa_local.data(), local_n, MPI_INT, all_sa.data(), counts.data(), displs.data(), MPI_INT, 0,
                MPI_COMM_WORLD);

    // Rank 0 merges and builds LCP + finds LRS
    if (rank == 0)
    {
        std::vector<int> sa_global;
        merge_k_sorted_lists(text, all_sa, counts, sa_global);

        std::vector<int> rk(n), lcp(n);
        build_lcp(text, sa_global, rk, lcp);

        int max_lcp = 0, pos = 0;

        for (int i = 1; i < static_cast<int>(n); ++i)
        {
            if (lcp[i] > max_lcp)
            {
                max_lcp = lcp[i];
                pos = sa_global[i];
            }
        }

        // Gather build times
        std::vector<double> times(size);
        MPI_Gather(&time_build, 1, MPI_DOUBLE, times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Print results
        std::cout << "\n=== MPI Suffix-Array + LCP across " << size << " ranks ===\n";

        for (int r = 0; r < size; ++r)
        {
            std::cout << "rank=" << r << " time_build=" << times[r] << " s\n";
        }

        std::cout << "global max_lrs_len=" << max_lcp << " pos=" << pos << "\n";
    }
    else
    {
        // Non-zero ranks still send their times
        MPI_Gather(&time_build, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
