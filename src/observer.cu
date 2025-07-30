// src/observer.cu

#include "observer.h"
#include "observer_kernel.cuh"
#include "state_manager.h"
#include "histogram_kernel.cuh"
#include <cuda_runtime.h>               // cudaMalloc, cudaFree, cudaMemcpyAsync, cudaStreamSynchronize
#include <algorithm>
#include <limits>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>

namespace MyQuantLib {

// ────────────────────────────────────────────────────────────────────────────
// MovingAverageObserverKernel_CPU
// ────────────────────────────────────────────────────────────────────────────
MovingAverageObserverKernel_CPU::MovingAverageObserverKernel_CPU(
	const OrtApi& api, const OrtKernelInfo* info) {
	Ort::ConstKernelInfo k(info);
	// 1) momentum 읽기 (기존 로직 그대로)
	momentum_ = k.GetAttribute<float>("momentum");
	node_name_ = k.GetNodeName();
	// 3) StateManager 에 node_name_ 로 등록
	StateManager::get_instance().register_moving_average(node_name_);
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

	ObserverState* state = StateManager::get_instance().get_state_ptr(node_name_);

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
	const OrtApi& api, const OrtKernelInfo* info) {
	Ort::ConstKernelInfo k(info);
	momentum_ = k.GetAttribute<float>("momentum");
	node_name_ = k.GetNodeName();
	StateManager::get_instance().register_moving_average(node_name_);
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

  ObserverState* state = StateManager::get_instance().get_state_ptr(node_name_);
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
	node_name_ = k.GetNodeName();
	StateManager::get_instance().register_histogram(node_name_, bins_);
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
	int64_t idx;
	if (range > 0.f) {
	  idx = static_cast<int64_t>((X[i] - min_val) / bin_width);
	  idx = std::clamp(idx, int64_t(0), bins_ - 1);
	} else {
	  idx = bins_ / 2;
	}
	hist_data[idx]++;
  }

  // 3) output: identity
  auto out = ctx.GetOutput(0, shape.data(), shape.size());
  float* Y = out.GetTensorMutableData<float>();
  std::copy(X, X + N, Y);

  // 4) StateManager에 기록
  ObserverState* st = StateManager::get_instance().get_state_ptr(node_name_);
  st->hist = std::move(hist_data);
  st->min  = min_val;
  st->max  = max_val;
}

// ────────────────────────────────────────────────────────────────────────────
// HistogramObserverKernel_CUDA
// ────────────────────────────────────────────────────────────────────────────
HistogramObserverKernel_CUDA::HistogramObserverKernel_CUDA(
	const OrtApi& api, const OrtKernelInfo* info) {
	Ort::ConstKernelInfo k(info);
	bins_ = k.GetAttribute<int64_t>("bins");
	node_name_ = k.GetNodeName();
	StateManager::get_instance().register_histogram(node_name_, bins_);
}

void HistogramObserverKernel_CUDA::Compute(OrtKernelContext* context) {
	
  Ort::KernelContext ctx(context);

  // 1) 입력 읽기
  auto input = ctx.GetInput(0);
  auto info  = input.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  int64_t N  = info.GetElementCount();
  const float* X = input.GetTensorData<float>();

  // 2) CUDA 스트림 및 출력(identity) 준비
  cudaStream_t stream =
	  reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());
  auto out = ctx.GetOutput(0, shape.data(), shape.size());
  float* Y = out.GetTensorMutableData<float>();

  // 3) StateManager가 초기화 시 할당해둔 디바이스 버퍼 꺼내기
  ObserverState* st = StateManager::get_instance().get_state_ptr(node_name_);
  int64_t*      dH = reinterpret_cast<int64_t*>(st->device_hist_buffer);

  // 4) 현재 입력(X)의 min/max 값 계산
  thrust::device_ptr<const float> dptr(X);
  auto mm = thrust::minmax_element(thrust::cuda::par.on(stream), dptr, dptr + N);
  float update_min = *mm.first;
  float update_max = *mm.second;

  // 5) 누적 min/max 값 업데이트 (개선된 버전)
  st->min = std::min(st->min, update_min);
  st->max = std::max(st->max, update_max);

  // 6) 업데이트된 누적 min/max 값으로 히스토그램 계산 및 identity 복사
  launch_histogram_cuda(X, dH, Y, N, bins_, st->min, st->max, stream);

  // 7) 디바이스 -> 호스트로 히스토그램 데이터 복사
  std::vector<int64_t> hist_data(bins_);
  cudaMemcpyAsync(hist_data.data(),
				  dH,
				  bins_ * sizeof(int64_t),
				  cudaMemcpyDeviceToHost,
				  stream);

  // 8) 스트림 동기화 (모든 CUDA 작업이 끝날 때까지 대기)
  cudaStreamSynchronize(stream);

  // 9) StateManager에 최종 결과 기록
  // st->min, st->max는 이미 위에서 업데이트됨
  st->hist = std::move(hist_data);
}

}  // namespace MyQuantLib
