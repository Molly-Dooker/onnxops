// src/histogram_kernel.cuh
#pragma once

#include <cstdint>
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

/**
 * Launches a CUDA histogram + identity kernel:
 *  - computes min/max over X
 *  - bins values into H array of length 'bins'
 *  - copies X into Y
 *
 * @param X      Device pointer to input floats
 * @param H      Device pointer to int64 histogram array (length 'bins')
 * @param Y      Device pointer to output identity floats (length N)
 * @param N      Number of elements in X and Y
 * @param bins   Number of histogram bins
 * @param stream CUDA stream to launch kernels on
 */
void launch_histogram_cuda(const float* X,
                           int64_t* H,
                           float* Y,
                           int64_t N,
                           int64_t bins,
                           float min_val,       // min_val 인자 추가
                           float max_val,       // max_val 인자 추가
                           cudaStream_t stream);