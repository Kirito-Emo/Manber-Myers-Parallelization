// Created by Emanuele (https://github.com/Kirito-Emo)

#pragma once
#include <cuda_runtime.h>

/**
 * Initializes a custom asynchronous memory pool for the current device.
 * This allows memory to be reused and allocated via cudaMallocAsync.
 * @param pool Pointer to a cudaMemPool_t that will be created and configured.
 */
void cuda_mem_pool_init(cudaMemPool_t *pool);

/**
 * Allocates memory asynchronously from the given memory pool.
 * @param pool The memory pool from which to allocate
 * @param size The size in bytes
 * @param stream CUDA stream (default: 0)
 * @return Pointer to the allocated device memory
 */
void *cuda_malloc_async(cudaMemPool_t pool, size_t size, cudaStream_t stream = 0);

/**
 * Frees memory asynchronously back to the pool.
 * @param ptr Pointer to device memory
 * @param stream CUDA stream (default: 0)
 */
void cuda_free_async(void *ptr, cudaStream_t stream = 0);

/**
 * Destroys the memory pool and releases all resources.
 * NOTE: Only call this if you used a custom-created pool (not the default one).
 * @param pool The memory pool to destroy
 */
void cuda_mem_pool_destroy(cudaMemPool_t pool);
