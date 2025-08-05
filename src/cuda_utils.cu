// Created by Emanuele (https://github.com/Kirito-Emo)

#include "cuda_utils.h"
#include <iostream>

void printCudaInfo()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    std::cout << "Found " << deviceCount << " CUDA device(s):\n";

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "  [" << i << "] " << prop.name
                  << " - " << (prop.totalGlobalMem >> 20) << " MB global mem"
                  << ", " << prop.multiProcessorCount << " SMs\n";
    }
}
