// Created by Emanuele (https://github.com/Kirito-Emo)

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "cuda_utils.h"
#include "cuda_malloc_utils.h"
#include "suffix_array_cuda_parallel.h"
#include <iomanip>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <size-in-MB>\n";
        return 1;
    }

    // Parse size and file
    int mb = std::stoi(argv[1]);
    if (mb != 1 && mb != 50 && mb != 100 && mb != 200 && mb != 500)
    {
        std::cerr << "Error: allowed sizes are 1, 50, 100, 200, 500 MB.\n";
        return 1;
    }

    size_t n = static_cast<size_t>(mb) * 1024 * 1024;
    std::string path = "../random_strings/string_" + std::to_string(mb) + "MB.bin";

    std::ifstream fin(path, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Error: failed to open " << path << "\n";
        return 1;
    }

    std::vector<uint8_t> text(n);
    fin.read(reinterpret_cast<char *>(text.data()), n);
    fin.close();

    std::vector<int> sa(n);
    int lrs_len = 0, lrs_pos = 0;

    // Init CUDA memory pool
    cudaMemPool_t pool;
    cuda_mem_pool_init(&pool);

    // Compute dynamic number of streams
    constexpr int MAX_STREAMS = 8;
    constexpr int MIN_MB_PER_STREAM = 10;
    int num_streams = std::min(MAX_STREAMS, std::max(1, mb / MIN_MB_PER_STREAM));

    auto t0 = std::chrono::high_resolution_clock::now();

    build_suffix_array_cuda_parallel(text, sa, num_streams, pool);
    find_lrs_cuda_parallel(text, sa, lrs_pos, lrs_len, pool);

    auto t1 = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "=== CUDA PARALLEL SA + LCP ===\n";
    std::cout << "size=" << mb << " MB  streams=" << num_streams << "   time_build=" << time_sec << " s\n";
    std::cout << "max_lrs_len=" << lrs_len << "   pos=" << lrs_pos << "\n";
    if (lrs_len > 0)
    {
        std::cout << "LRS (hex): ";
        for (int i = 0; i < lrs_len; ++i)
            std::cout << std::hex << std::uppercase << std::setw(2)
                      << std::setfill('0') << (int)text[lrs_pos + i] << ' ';
        std::cout << std::dec << "\n";
    }
    else
        std::cout << "LRS: (no repeated substring found)\n";

    printCudaInfo();

    cuda_suffix_array_cleanup();
    cuda_mem_pool_destroy(pool);

    return 0;
}
