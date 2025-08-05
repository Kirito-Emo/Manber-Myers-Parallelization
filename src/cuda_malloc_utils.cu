// Created by Emanuele (https://github.com/Kirito-Emo)

#include "cuda_malloc_utils.h"
#include "cuda_utils.h"
#include <cuda_runtime_api.h>

// Initializes a custom asynchronous memory pool for reuse across allocations
void cuda_mem_pool_init(cudaMemPool_t* pool)
{
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    // Create a new memory pool instead of using the default one
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;

    CUDA_CHECK(cudaMemPoolCreate(pool, &props));

    // Set optional attributes
    size_t threshold = 1ULL << 32; // 4 GB threshold before releasing memory
    CUDA_CHECK(cudaMemPoolSetAttribute(*pool, cudaMemPoolAttrReleaseThreshold, &threshold));

    // Set the pool as active for the current device (optional)
    CUDA_CHECK(cudaDeviceSetMemPool(device, *pool));
}

// Allocates memory from the given pool
void* cuda_malloc_async(cudaMemPool_t pool, size_t size, cudaStream_t stream)
{
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocFromPoolAsync(&ptr, size, pool, stream));
    return ptr;
}

// Frees memory asynchronously
void cuda_free_async(void* ptr, cudaStream_t stream)
{
    CUDA_CHECK(cudaFreeAsync(ptr, stream));
}

// Destroys the memory pool if custom
void cuda_mem_pool_destroy(cudaMemPool_t pool)
{
    // Only destroy if the pool is not the default one
    CUDA_CHECK(cudaMemPoolDestroy(pool));
}
