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
  int64_t b = 0;
  if (bin_width > 0.f) {
    b = static_cast<int64_t>((v - min_val) / bin_width);
    if (b < 0) b = 0;
    else if (b >= bins) b = bins - 1;
  }
  // int64_t → unsigned long long 로 캐스트
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
                           cudaStream_t stream) {
  // 1) find min/max via thrust
  thrust::device_ptr<const float> dptr(X);
  auto mm = thrust::minmax_element(thrust::cuda::par.on(stream), dptr, dptr + N);
  float min_val = *mm.first;
  float max_val = *mm.second;
  float range = max_val - min_val;
  float bin_width = (bins > 0) ? (range / static_cast<float>(bins)) : 0.f;

  // 2) zero histogram
  cudaMemsetAsync(H, 0, bins * sizeof(int64_t), stream);

  // 3) launch HistKernel and IdentityKernel
  int threads = 256;
  int blocks = static_cast<int>((N + threads - 1) / threads);
  HistKernel<<<blocks, threads, 0, stream>>>(X, H, N, min_val, bin_width, bins);
  IdentityKernel<<<blocks, threads, 0, stream>>>(X, Y, N);

  // 4) (필요시) 스트림 동기화
  cudaStreamSynchronize(stream);
}
