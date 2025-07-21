#pragma once

#include "state_manager.h"

// cuda_runtime.h를 포함하는 대신 cudaStream_t를 전방 선언합니다.
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

void launch_observer_kernel(
    const float* input,
    float* output,
    long long num_elements,
    MyQuantLib::ObserverState* state,
    float momentum,
    cudaStream_t stream
);