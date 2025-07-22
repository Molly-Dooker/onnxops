// observer.cpp
#include "observer.h"
#include "observer_kernel.cuh"
#include "state_manager.h"
#include <algorithm>
#include <limits>

namespace MyQuantLib {

// ─ CPU Kernel ────────────────────────────────────────────────────────────────
MovingAverageObserverKernel_CPU::MovingAverageObserverKernel_CPU(
    const OrtApi&, const OrtKernelInfo* info) {
  Ort::ConstKernelInfo k(info);
  momentum_ = k.GetAttribute<float>("momentum");
  id_       = k.GetAttribute<std::string>("id");
  StateManager::get_instance().register_observer(id_);
}

void MovingAverageObserverKernel_CPU::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto input_value = ctx.GetInput(0);
  auto input_info = input_value.GetTensorTypeAndShapeInfo();
  auto shape = input_info.GetShape();
  int64_t N = input_info.GetElementCount();
  const float* X = input_value.GetTensorData<float>();

  auto output_value = ctx.GetOutput(0, shape.data(), shape.size());
  float* Y = output_value.GetTensorMutableData<float>();

  ObserverState* state = StateManager::get_instance().get_state_ptr(id_);

  float batch_min = std::numeric_limits<float>::max();
  float batch_max = std::numeric_limits<float>::lowest();
  for (int64_t i = 0; i < N; ++i) {
    float v = X[i];
    if (v < batch_min) batch_min = v;
    if (v > batch_max) batch_max = v;
  }
  if (state->min == std::numeric_limits<float>::max()) {
    state->min = batch_min;
    state->max = batch_max;
  } else {
    state->min = state->min * momentum_ + batch_min * (1.0f - momentum_);
    state->max = state->max * momentum_ + batch_max * (1.0f - momentum_);
  }
  std::copy_n(X, N, Y);
}

// ─ GPU Kernel ────────────────────────────────────────────────────────────────
MovingAverageObserverKernel_CUDA::MovingAverageObserverKernel_CUDA(
    const OrtApi&, const OrtKernelInfo* info) {
  Ort::ConstKernelInfo k(info);
  momentum_ = k.GetAttribute<float>("momentum");
  id_       = k.GetAttribute<std::string>("id");
  StateManager::get_instance().register_observer(id_);
}

void MovingAverageObserverKernel_CUDA::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto input_value = ctx.GetInput(0);
  auto input_info = input_value.GetTensorTypeAndShapeInfo();
  auto shape = input_info.GetShape();
  int64_t N = input_info.GetElementCount();
  const float* X = input_value.GetTensorData<float>();

  auto output_value = ctx.GetOutput(0, shape.data(), shape.size());
  float* Y = output_value.GetTensorMutableData<float>();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());

  ObserverState* state = StateManager::get_instance().get_state_ptr(id_);
  launch_observer_kernel(X, Y, N, state, momentum_, stream);
}

// ─ CustomOpBase 메서드 정의 ─────────────────────────────────────────────────
// CPU용 Op
void* MovingAverageObserverOp_CPU::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return new MovingAverageObserverKernel_CPU(api, info);
}
const char* MovingAverageObserverOp_CPU::GetName() const { return "MovingAverageObserver"; }
const char* MovingAverageObserverOp_CPU::GetExecutionProviderType() const { return "CPUExecutionProvider"; }
size_t MovingAverageObserverOp_CPU::GetInputTypeCount() const { return 1; }
ONNXTensorElementDataType MovingAverageObserverOp_CPU::GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
size_t MovingAverageObserverOp_CPU::GetOutputTypeCount() const { return 1; }
ONNXTensorElementDataType MovingAverageObserverOp_CPU::GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

// CUDA용 Op
void* MovingAverageObserverOp_CUDA::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return new MovingAverageObserverKernel_CUDA(api, info);
}
const char* MovingAverageObserverOp_CUDA::GetName() const { return "MovingAverageObserver"; }
const char* MovingAverageObserverOp_CUDA::GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
size_t MovingAverageObserverOp_CUDA::GetInputTypeCount() const { return 1; }
ONNXTensorElementDataType MovingAverageObserverOp_CUDA::GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
size_t MovingAverageObserverOp_CUDA::GetOutputTypeCount() const { return 1; }
ONNXTensorElementDataType MovingAverageObserverOp_CUDA::GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
}