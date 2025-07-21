#include "observer_kernel.cuh"
#include <cuda_runtime.h> //.cu 파일에서는 CUDA 헤더를 포함합니다.
#include <float.h>

// float 타입을 위한 atomicMin/Max 헬퍼 함수 구현
__device__ static void atomicMinFloat(float* addr, float value) {
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old_val_uint = *addr_as_uint;
    unsigned int new_val_uint = __float_as_uint(value);

    while (value < __uint_as_float(old_val_uint)) {
        unsigned int assumed_val = old_val_uint;
        old_val_uint = atomicCAS(addr_as_uint, assumed_val, new_val_uint);
        if (old_val_uint == assumed_val) {
            break;
        }
    }
}

__device__ static void atomicMaxFloat(float* addr, float value) {
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old_val_uint = *addr_as_uint;
    unsigned int new_val_uint = __float_as_uint(value);

    while (value > __uint_as_float(old_val_uint)) {
        unsigned int assumed_val = old_val_uint;
        old_val_uint = atomicCAS(addr_as_uint, assumed_val, new_val_uint);
        if (old_val_uint == assumed_val) {
            break;
        }
    }
}


__global__ void minmax_kernel(const float* input, long long n, float* batch_min, float* batch_max) {
    // [수정] 동적 공유 메모리를 올바르게 배열로 선언
    extern __shared__ float s_mem[];
    
    unsigned int tid = threadIdx.x;
    // [수정] 공유 메모리 포인터 설정
    float* s_min = s_mem;
    float* s_max = s_mem + blockDim.x; 

    unsigned int i = blockIdx.x * blockDim.x + tid;

    float my_min = (i < n)? input[i] : FLT_MAX;
    float my_max = (i < n)? input[i] : -FLT_MAX;

    for (i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
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
        // [수정] 포인터가 아닌, 리덕션이 완료된 값을 전달
        atomicMinFloat(batch_min, s_min[0]);
        atomicMaxFloat(batch_max, s_max[0]);
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
    // min/max 두 배열을 위한 공유 메모리 크기
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);

    minmax_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(input, num_elements, d_batch_min, d_batch_max);

    update_state_and_copy<<<blocks_per_grid, threads_per_block, 0, stream>>>(input, output, num_elements, d_batch_min, d_batch_max, state, momentum);

    cudaFreeAsync(d_batch_min, stream);
    cudaFreeAsync(d_batch_max, stream);
}