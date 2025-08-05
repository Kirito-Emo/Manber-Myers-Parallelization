// Created by Emanuele (https://github.com/Kirito-Emo)

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "cuda_utils.h"
#include "cuda_malloc_utils.h"
#include "suffix_array_cuda.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <size-in-MB>\n";
        return 1;
    }

    // Parse size and validate
    int mb = std::stoi(argv[1]);
    if (mb != 1 && mb != 50 && mb != 100 && mb != 200 && mb != 500)
    {
        std::cerr << "Error: allowed sizes are 1, 50, 100, 200, 500 MB.\n";
        return 1;
    }

    size_t n = static_cast<size_t>(mb) * 1024 * 1024;
    std::string path = "../random_strings/string_" + std::to_string(mb) + "MB.bin";

    // Load file
    std::ifstream fin(path, std::ios::binary);
    if (!fin)
    {
        std::cerr << "Error: failed to open " << path << "\n";
        return 1;
    }

    std::vector<uint8_t> text(n);
    fin.read(reinterpret_cast<char *>(text.data()), n);
    fin.close();

    // Allocate output buffers
    std::vector<int> sa(n);
    int lrs_len = 0, lrs_pos = 0;

    // Initialize CUDA memory pool
    cudaMemPool_t pool;
    cuda_mem_pool_init(&pool);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Full pipeline on GPU: SA + LRS
    build_suffix_array_cuda(text, sa, pool);
    find_lrs_cuda(text, sa, lrs_pos, lrs_len, pool);

    auto t1 = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "=== CUDA Suffix Array + LCP ===\n";
    std::cout << "size=" << mb << " MB   time_build=" << time_sec << " s\n";
    std::cout << "max_lrs_len=" << lrs_len << "   pos=" << lrs_pos << "\n";

    // GPU info
    printCudaInfo();

    // Cleanup
    cuda_suffix_array_cleanup();
    cuda_mem_pool_destroy(pool);

    return 0;
}
