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

// ─ Histogram Observer (New) ────────────────────────────────────────────────────
// CPU kernel: computes min/max, histc, returns histogram + identity
struct HistogramObserverKernel_CPU {
  HistogramObserverKernel_CPU(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  int64_t bins_;
  std::string id_;
};

// GPU kernel: same functionality on CUDA
struct HistogramObserverKernel_CUDA {
  HistogramObserverKernel_CUDA(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  int64_t bins_;
  std::string id_;
};

// CustomOp registration for CPU
struct HistogramObserverOp_CPU
    : Ort::CustomOpBase<HistogramObserverOp_CPU, HistogramObserverKernel_CPU> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new HistogramObserverKernel_CPU(api, info);
  }
  const char* GetName() const { return "HistogramObserver"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

// CustomOp registration for CUDA
struct HistogramObserverOp_CUDA
    : Ort::CustomOpBase<HistogramObserverOp_CUDA, HistogramObserverKernel_CUDA> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new HistogramObserverKernel_CUDA(api, info);
  }
  const char* GetName() const { return "HistogramObserver"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};


} // namespace MyQuantLib