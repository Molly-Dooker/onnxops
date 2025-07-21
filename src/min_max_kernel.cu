#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "min_max_kernel.h"

struct Float2 { float min_val; float max_val; };

__device__ Float2 merge(Float2 a, Float2 b) {
    return {min(a.min_val, b.min_val), max(a.max_val, b.max_val)};
}

__global__ void min_max_kernel(const float* input, size_t n, Float2* output) {
    extern __shared__ Float2 sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = {(i < n) ? input[i] : FLT_MAX, (i < n) ? input[i] : -FLT_MAX};
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = merge(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void launch_min_max_kernel(const float* input_gpu, size_t count, float* output_gpu) {
    int threads_per_block = 256;
    int blocks_per_grid = (count + threads_per_block - 1) / threads_per_block;

    Float2* d_intermediate_results = nullptr;
    cudaMalloc(&d_intermediate_results, blocks_per_grid * sizeof(Float2));

    min_max_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(Float2)>>>(
        input_gpu, count, d_intermediate_results);

    if (blocks_per_grid > 1) {
        min_max_kernel<<<1, threads_per_block, threads_per_block * sizeof(Float2)>>>(
            (const float*)d_intermediate_results, blocks_per_grid * 2, (Float2*)output_gpu);
    } else {
        cudaMemcpy(output_gpu, d_intermediate_results, sizeof(Float2), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_intermediate_results);
}