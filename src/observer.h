// observer.h
#pragma once

#include "onnxruntime_cxx_api.h"
#include <string>

namespace MyQuantLib {

// ─ CPU 커널 ───────────────────────────────────────────────────────────────
struct MovingAverageObserverKernel_CPU {
  MovingAverageObserverKernel_CPU(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

private:
  float momentum_;
  std::string id_;
};

// ─ GPU 커널 ───────────────────────────────────────────────────────────────
struct MovingAverageObserverKernel_CUDA {
  MovingAverageObserverKernel_CUDA(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

private:
  float momentum_;
  std::string id_;
};

// ─ CustomOp 등록 (CPU 전용) ───────────────────────────────────────────────
struct MovingAverageObserverOp_CPU 
    : Ort::CustomOpBase<MovingAverageObserverOp_CPU, MovingAverageObserverKernel_CPU> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  const char* GetExecutionProviderType() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

// ─ CustomOp 등록 (CUDA 전용) ──────────────────────────────────────────────
struct MovingAverageObserverOp_CUDA 
    : Ort::CustomOpBase<MovingAverageObserverOp_CUDA, MovingAverageObserverKernel_CUDA> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  const char* GetExecutionProviderType() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

} // namespace MyQuantLib