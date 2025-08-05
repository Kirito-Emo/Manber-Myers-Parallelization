// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cuda_runtime.h>
#include <iostream>

// Error handling macro for CUDA API calls
#define CUDA_CHECK(ans)                                                                                                \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
        if (abort)
            exit(code);
    }
}

// Print GPU info
void printCudaInfo();
