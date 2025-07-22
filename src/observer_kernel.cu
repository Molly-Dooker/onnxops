#include "observer_kernel.cuh"
#include <cuda_runtime.h>
#include <float.h>

__device__ static void atomicMinFloat(float* addr, float value) {
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    unsigned int val_uint = __float_as_uint(value);
    // 원자적(atomic)으로 주소에 있는 값을 value와 비교하여 min 연산
    while (value < __uint_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_uint, assumed, val_uint);
        if (assumed == old) break;
    }
}
__device__ static void atomicMaxFloat(float* addr, float value) {
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    unsigned int val_uint = __float_as_uint(value);
    while (value > __uint_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_uint, assumed, val_uint);
        if (assumed == old) break;
    }
}

// 입력 배열에서 min과 max를 계산하는 CUDA 커널
__global__ void minmax_kernel(const float* input, long long n, float* batch_min, float* batch_max) {
    // 동적 공유 메모리 영역: [0..blockDim.x-1] -> s_min, [blockDim.x..2*blockDim.x-1] -> s_max
    extern __shared__ float s_mem[];
    float* s_min = s_mem;
    float* s_max = s_mem + blockDim.x;

    unsigned int tid = threadIdx.x;
    // 이 스레드가 담당할 초기 인덱스
    long long i = blockIdx.x * blockDim.x + tid;
    // 각 스레드별로 관찰한 최소/최대 초기값 설정
    float my_min = FLT_MAX;
    float my_max = -FLT_MAX;
    if (i < n) {
        my_min = input[i];
        my_max = input[i];
        // 동일 스레드가 담당하는 다른 요소들도 처리 (gridStrideLoop)
        for (i += blockDim.x * gridDim.x; i < n; i += blockDim.x * gridDim.x) {
            float val = input[i];
            if (val < my_min) my_min = val;
            if (val > my_max) my_max = val;
        }
    }
    // 스레드별 결과를 공유 메모리에 저장
    s_min[tid] = my_min;
    s_max[tid] = my_max;
    __syncthreads();

    // 공유 메모리 내 병렬 reduction을 통해 block별 min/max 계산
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }

    // 각 블록의 첫 번째 스레드가 해당 블록의 min/max를 전역 메모리로 원자적으로 기록
    if (tid == 0) {
        atomicMinFloat(batch_min, s_min[0]);
        atomicMaxFloat(batch_max, s_max[0]);
    }
}

// GPU 커널들을 실행하고 상태를 갱신하는 호스트 함수
void launch_observer_kernel(
    const float* input,
    float* output,
    long long num_elements,
    MyQuantLib::ObserverState* state,
    float momentum,
    cudaStream_t stream
) {
    // 1. 디바이스 메모리 할당 (현재 배치의 min/max 결과 저장용)
    float* d_batch_min;
    float* d_batch_max;
    cudaMalloc(&d_batch_min, sizeof(float));
    cudaMalloc(&d_batch_max, sizeof(float));
    // 초기값 설정: d_batch_min = +∞, d_batch_max = -∞
    float h_init_min = FLT_MAX;
    float h_init_max = -FLT_MAX;
    cudaMemcpyAsync(d_batch_min, &h_init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_batch_max, &h_init_max, sizeof(float), cudaMemcpyHostToDevice, stream);

    // 2. minmax_kernel 실행 – 입력 텐서 전체에 대한 min/max 계산
    int threads_per_block = 256;
    int blocks_per_grid = static_cast<int>(
        std::min( (num_elements + threads_per_block - 1) / threads_per_block, (long long)1024 )
    );
    size_t shared_mem_size = 2 * threads_per_block * sizeof(float);
    minmax_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(input, num_elements, d_batch_min, d_batch_max);

    // 3. 입력 텐서를 출력 버퍼로 복사 (Device->Device 메모리 복사)
    cudaMemcpyAsync(output, input, num_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // 4. GPU에서 계산된 현재 배치 min/max 값을 호스트로 복사하여 가져오기
    float h_batch_min, h_batch_max;
    cudaMemcpyAsync(&h_batch_min, d_batch_min, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_batch_max, d_batch_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    // GPU 연산 동기화 – 여기서 모든 CUDA 작업이 완료되길 대기
    cudaStreamSynchronize(stream);

    // 5. 호스트에서 상태 갱신 (모멘텀 적용)
    if (state->min == FLT_MAX) {
        // 첫 번째 업데이트라면 배치 통계를 그대로 반영
        state->min = h_batch_min;
        state->max = h_batch_max;
    } else {
        state->min = state->min * momentum + h_batch_min * (1.0f - momentum);
        state->max = state->max * momentum + h_batch_max * (1.0f - momentum);
    }

    // 6. 디바이스 임시 메모리 해제
    cudaFree(d_batch_min);
    cudaFree(d_batch_max);
}
