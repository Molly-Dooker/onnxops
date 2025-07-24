// src/observer.cpp

#include "observer.h"
#include "observer_kernel.cuh"
#include "state_manager.h"
#include "histogram_kernel.cuh"

#include <algorithm>
#include <limits>
#include <vector>

namespace MyQuantLib {

// ────────────────────────────────────────────────────────────────────────────
// MovingAverageObserverKernel_CPU
// ────────────────────────────────────────────────────────────────────────────
MovingAverageObserverKernel_CPU::MovingAverageObserverKernel_CPU(
    const OrtApi&, const OrtKernelInfo* info) {
  Ort::ConstKernelInfo k(info);
  momentum_ = k.GetAttribute<float>("momentum");
  id_       = k.GetAttribute<std::string>("id");
  StateManager::get_instance().register_moving_average(id_);
}

void MovingAverageObserverKernel_CPU::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto input      = ctx.GetInput(0);
  auto info       = input.GetTensorTypeAndShapeInfo();
  auto shape      = info.GetShape();
  int64_t N       = info.GetElementCount();
  const float* X  = input.GetTensorData<float>();

  auto out_value  = ctx.GetOutput(0, shape.data(), shape.size());
  float*       Y  = out_value.GetTensorMutableData<float>();

  ObserverState* state = StateManager::get_instance().get_state_ptr(id_);

  float batch_min = std::numeric_limits<float>::max();
  float batch_max = std::numeric_limits<float>::lowest();
  for (int64_t i = 0; i < N; ++i) {
    float v = X[i];
    batch_min = std::min(batch_min, v);
    batch_max = std::max(batch_max, v);
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

// ────────────────────────────────────────────────────────────────────────────
// MovingAverageObserverKernel_CUDA
// ────────────────────────────────────────────────────────────────────────────
MovingAverageObserverKernel_CUDA::MovingAverageObserverKernel_CUDA(
    const OrtApi&, const OrtKernelInfo* info) {
  Ort::ConstKernelInfo k(info);
  momentum_ = k.GetAttribute<float>("momentum");
  id_       = k.GetAttribute<std::string>("id");
  StateManager::get_instance().register_moving_average(id_);
}

void MovingAverageObserverKernel_CUDA::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto input      = ctx.GetInput(0);
  auto info       = input.GetTensorTypeAndShapeInfo();
  auto shape      = info.GetShape();
  int64_t N       = info.GetElementCount();
  const float* X  = input.GetTensorData<float>();

  auto out_value  = ctx.GetOutput(0, shape.data(), shape.size());
  float*       Y  = out_value.GetTensorMutableData<float>();

  cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());

  ObserverState* state = StateManager::get_instance().get_state_ptr(id_);
  launch_observer_kernel(X, Y, N, state, momentum_, stream);
}

// ────────────────────────────────────────────────────────────────────────────
// CustomOpBase 구현 (MovingAverage)
// ────────────────────────────────────────────────────────────────────────────
void* MovingAverageObserverOp_CPU::CreateKernel(const OrtApi& api,
                                                const OrtKernelInfo* info) const {
  return new MovingAverageObserverKernel_CPU(api, info);
}
const char* MovingAverageObserverOp_CPU::GetName() const {
  return "MovingAverageObserver";
}
const char* MovingAverageObserverOp_CPU::GetExecutionProviderType() const {
  return "CPUExecutionProvider";
}
size_t MovingAverageObserverOp_CPU::GetInputTypeCount() const { return 1; }
ONNXTensorElementDataType
MovingAverageObserverOp_CPU::GetInputType(size_t) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
size_t MovingAverageObserverOp_CPU::GetOutputTypeCount() const { return 1; }
ONNXTensorElementDataType
MovingAverageObserverOp_CPU::GetOutputType(size_t) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

void* MovingAverageObserverOp_CUDA::CreateKernel(const OrtApi& api,
                                                 const OrtKernelInfo* info) const {
  return new MovingAverageObserverKernel_CUDA(api, info);
}
const char* MovingAverageObserverOp_CUDA::GetName() const {
  return "MovingAverageObserver";
}
const char* MovingAverageObserverOp_CUDA::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}
size_t MovingAverageObserverOp_CUDA::GetInputTypeCount() const { return 1; }
ONNXTensorElementDataType
MovingAverageObserverOp_CUDA::GetInputType(size_t) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
size_t MovingAverageObserverOp_CUDA::GetOutputTypeCount() const { return 1; }
ONNXTensorElementDataType
MovingAverageObserverOp_CUDA::GetOutputType(size_t) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

// ────────────────────────────────────────────────────────────────────────────
// HistogramObserverKernel_CPU
// ────────────────────────────────────────────────────────────────────────────
HistogramObserverKernel_CPU::HistogramObserverKernel_CPU(
    const OrtApi& api, const OrtKernelInfo* info) {
  Ort::ConstKernelInfo k(info);
  bins_ = k.GetAttribute<int64_t>("bins");
  id_   = k.GetAttribute<std::string>("id");
  StateManager::get_instance().register_histogram(id_, bins_);
}

void HistogramObserverKernel_CPU::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto input      = ctx.GetInput(0);
  auto info       = input.GetTensorTypeAndShapeInfo();
  auto shape      = info.GetShape();
  int64_t N       = info.GetElementCount();
  const float* X  = input.GetTensorData<float>();

  // 1) min/max 계산
  float min_val = std::numeric_limits<float>::infinity();
  float max_val = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < N; ++i) {
    min_val = std::min(min_val, X[i]);
    max_val = std::max(max_val, X[i]);
  }
  float range     = max_val - min_val;
  float bin_width = range / static_cast<float>(bins_);

  // 2) histogram 카운트
  std::vector<int64_t> hist_data(bins_, 0);
  for (int64_t i = 0; i < N; ++i) {
    int64_t idx = (range > 0.f)
                      ? static_cast<int64_t>((X[i] - min_val) / bin_width)
                      : 0;
    if (idx == bins_) idx = bins_ - 1;
    hist_data[idx]++;
  }

  // output0: histogram
  auto out0 = ctx.GetOutput(0, &bins_, 1);
  int64_t* H    = out0.GetTensorMutableData<int64_t>();
  std::copy(hist_data.begin(), hist_data.end(), H);

  // output1: identity
  auto out1 = ctx.GetOutput(1, shape.data(), shape.size());
  float*   Y    = out1.GetTensorMutableData<float>();
  std::copy(X, X + N, Y);

  // 상태 매니저에 복사
  ObserverState* st = StateManager::get_instance().get_state_ptr(id_);
  st->hist = hist_data;
}

// ────────────────────────────────────────────────────────────────────────────
// HistogramObserverKernel_CUDA
// ────────────────────────────────────────────────────────────────────────────
HistogramObserverKernel_CUDA::HistogramObserverKernel_CUDA(
    const OrtApi& api, const OrtKernelInfo* info) {
  Ort::ConstKernelInfo k(info);
  bins_ = k.GetAttribute<int64_t>("bins");
  id_   = k.GetAttribute<std::string>("id");
  StateManager::get_instance().register_histogram(id_, bins_);
}

void HistogramObserverKernel_CUDA::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto input     = ctx.GetInput(0);
  auto info      = input.GetTensorTypeAndShapeInfo();
  auto shape     = info.GetShape();
  int64_t N      = info.GetElementCount();
  const float* X = input.GetTensorData<float>();

  // GPU 스트림 forward-declared in observer_kernel.cuh
  cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());

  // output 준비
  auto out0 = ctx.GetOutput(0, &bins_, 1);
  auto out1 = ctx.GetOutput(1, shape.data(), shape.size());
  int64_t*    H = out0.GetTensorMutableData<int64_t>();
  float*      Y = out1.GetTensorMutableData<float>();

  // CUDA histogram + identity 실행
  launch_histogram_cuda(X, H, Y, N, bins_, stream);
}

}  // namespace MyQuantLib
