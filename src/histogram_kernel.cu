// src/histogram_kernel.cu
#include "histogram_kernel.cuh"
#include <algorithm>
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cuda_runtime.h>  // .cu 에서는 cuda_runtime 포함 OK

// CUDA kernel: count into histogram bins
__global__ void HistKernel(const float* __restrict__ X,
                           int64_t* __restrict__ H,
                           int64_t N,
                           float min_val,
                           float bin_width,
                           int64_t bins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float v = X[idx];
  int64_t b;
  if (bin_width > 0.f) {
    b = static_cast<int64_t>((v - min_val) / bin_width);
    if (b < 0)        b = 0;
    else if (b >= bins) b = bins - 1;
  } else {
    b = bins / 2;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(&H[b]), 1ULL);
}

// CUDA kernel: identity copy
__global__ void IdentityKernel(const float* __restrict__ X,
                               float* __restrict__ Y,
                               int64_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    Y[idx] = X[idx];
  }
}

void launch_histogram_cuda(const float* X,
                           int64_t* H,
                           float* Y,
                           int64_t N,
                           int64_t bins,
                           float min_val,       // 인자로 받은 min_val
                           float max_val,       // 인자로 받은 max_val
                           cudaStream_t stream) {
  // 1) min/max 계산 로직 제거하고, 전달받은 값으로 bin_width 계산
  float range = max_val - min_val;
  float bin_width = (bins > 0 && range > 0) ? (range / static_cast<float>(bins)) : 0.f;

  // 2) 히스토그램 버퍼 초기화
  cudaMemsetAsync(H, 0, bins * sizeof(int64_t), stream);

  // 3) HistKernel 및 IdentityKernel 실행
  int threads = 256;
  int blocks = static_cast<int>((N + threads - 1) / threads);
  HistKernel<<<blocks, threads, 0, stream>>>(X, H, N, min_val, bin_width, bins);
  IdentityKernel<<<blocks, threads, 0, stream>>>(X, Y, N);

  // 4) 스트림 동기화는 호출하는 쪽에서 관리하므로 여기서 제거해도 무방
  cudaStreamSynchronize(stream);
}
