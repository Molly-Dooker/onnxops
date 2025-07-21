#include "observer_kernel.cuh"
#include <float.h>

__global__ void minmax_kernel(const float* input, long long n, float* batch_min, float* batch_max) {
    extern __shared__ float sdata;
    float* s_min = sdata;
    float* s_max = (float*)&sdata;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float my_min = (i < n)? input[i] : FLT_MAX;
    float my_max = (i < n)? input[i] : -FLT_MAX;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        my_min = fminf(my_min, input[i]);
        my_max = fmaxf(my_max, input[i]);
    }

    s_min[tid] = my_min;
    s_max[tid] = my_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(batch_min, s_min);
        atomicMax(batch_max, s_max);
    }
}

__global__ void update_state_and_copy(const float* input, float* output, long long n,
                                      const float* batch_min, const float* batch_max,
                                      MyQuantLib::ObserverState* state,
                                      float momentum) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (state->min == FLT_MAX) {
            state->min = *batch_min;
            state->max = *batch_max;
        } else {
            state->min = state->min * momentum + *batch_min * (1.0f - momentum);
            state->max = state->max * momentum + *batch_max * (1.0f - momentum);
        }
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        output[i] = input[i];
    }
}

void launch_observer_kernel(
    const float* input,
    float* output,
    long long num_elements,
    MyQuantLib::ObserverState* state,
    float momentum,
    cudaStream_t stream
) {
    float* d_batch_min;
    float* d_batch_max;
    cudaMalloc(&d_batch_min, sizeof(float));
    cudaMalloc(&d_batch_max, sizeof(float));

    float h_init_min = FLT_MAX;
    float h_init_max = -FLT_MAX;
    cudaMemcpyAsync(d_batch_min, &h_init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_batch_max, &h_init_max, sizeof(float), cudaMemcpyHostToDevice, stream);

    int threads_per_block = 256;
    int blocks_per_grid = min((int)((num_elements + threads_per_block - 1) / threads_per_block), 1024);
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);

    minmax_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(input, num_elements, d_batch_min, d_batch_max);

    update_state_and_copy<<<blocks_per_grid, threads_per_block, 0, stream>>>(input, output, num_elements, d_batch_min, d_batch_max, state, momentum);

    cudaFreeAsync(d_batch_min, stream);
    cudaFreeAsync(d_batch_max, stream);
}