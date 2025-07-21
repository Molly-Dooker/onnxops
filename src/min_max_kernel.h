#pragma once

#include <cuda_runtime.h>

// CUDA 커널을 호출하는 C++ 래퍼(wrapper) 함수
// 이 함수는 C++ 코드(.cpp)에서 호출할 수 있습니다.
void launch_min_max_kernel(
    const float* input_gpu,  // 입력 데이터 (GPU 포인터)
    size_t count,            // 데이터 개수
    float* output_gpu        // 결과 (min, max)를 저장할 GPU 포인터
);