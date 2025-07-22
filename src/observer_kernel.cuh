#pragma once

#include "state_manager.h"

// CUDA 스트림 타입 전방 선언 (cuda_runtime_api 대신 간략 선언)
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

// MovingAverageObserver CUDA 커널들을 실행하는 함수 선언
void launch_observer_kernel(
    const float* input,
    float* output,
    long long num_elements,
    MyQuantLib::ObserverState* state,
    float momentum,
    cudaStream_t stream
);
