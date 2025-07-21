#pragma once
#include <cuda_runtime.h>

void launch_min_max_kernel(
    const float* input_gpu,
    size_t count,
    float* output_gpu
);