#pragma once

#include <cuda_runtime.h>
#include "state_manager.h"

void launch_observer_kernel(
    const float* input,
    float* output,
    long long num_elements,
    MyQuantLib::ObserverState* state,
    float momentum,
    cudaStream_t stream
);