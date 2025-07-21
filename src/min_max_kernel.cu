#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "min_max_kernel.h"

// min/max 값을 함께 다루기 위한 구조체
struct Float2 {
    float min_val;
    float max_val;
};

// 두 Float2 구조체를 병합하는 함수 (GPU에서 실행)
__device__ Float2 merge(Float2 a, Float2 b) {
    Float2 c;
    c.min_val = min(a.min_val, b.min_val);
    c.max_val = max(a.max_val, b.max_val);
    return c;
}

// 병렬 리덕션을 수행하는 CUDA 커널
__global__ void min_max_kernel(const float* input, size_t n, Float2* output) {
    // 공유 메모리를 사용하여 블록 내 리덕션 수행
    extern __shared__ Float2 sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 각 스레드가 자신의 데이터를 로드하여 초기화
    Float2 my_val = { (i < n) ? input[i] : FLT_MAX, (i < n) ? input[i] : -FLT_MAX };
    sdata[tid] = my_val;
    __syncthreads();

    // 블록 내에서 리덕션 수행
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) {
            sdata[tid] = merge(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 블록의 최종 결과를 전역 메모리에 기록 (0번 스레드만)
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}


void launch_min_max_kernel(const float* input_gpu, size_t count, float* output_gpu) {
    int threads_per_block = 256;
    // 필요한 블록의 개수 계산
    int blocks_per_grid = (count + threads_per_block - 1) / threads_per_block;

    // 각 블록의 결과(min/max)를 저장할 임시 GPU 메모리 할당
    Float2* d_intermediate_results = nullptr;
    cudaMalloc(&d_intermediate_results, blocks_per_grid * sizeof(Float2));

    // 커널 실행
    min_max_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(Float2)>>>(
        input_gpu, count, d_intermediate_results);

    // 블록이 하나만 있었다면 바로 결과 복사
    if (blocks_per_grid == 1) {
         cudaMemcpy(output_gpu, d_intermediate_results, sizeof(Float2), cudaMemcpyDeviceToDevice);
    } else {
        // 여러 블록의 결과를 최종적으로 CPU에서 리덕션 (간단한 구현)
        // 또는 2차 커널을 실행하여 GPU에서 최종 리덕션 수행 가능
        std::vector<Float2> h_intermediate_results(blocks_per_grid);
        cudaMemcpy(h_intermediate_results.data(), d_intermediate_results, blocks_per_grid * sizeof(Float2), cudaMemcpyDeviceToHost);

        Float2 final_result = h_intermediate_results[0];
        for (int i = 1; i < blocks_per_grid; ++i) {
            final_result = merge(final_result, h_intermediate_results[i]);
        }
        cudaMemcpy(output_gpu, &final_result, sizeof(Float2), cudaMemcpyHostToDevice);
    }

    cudaFree(d_intermediate_results);
}