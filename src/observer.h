#pragma once

#include "onnxruntime_cxx_api.h"
#include "cuda_provider_factory.h"
#include <string>

namespace MyQuantLib {

struct MovingAverageObserverKernel {
    MovingAverageObserverKernel(const OrtApi& api, const OrtKernelInfo* info);
    void Compute(OrtKernelContext* context);

private:
    float momentum_;
    std::string id_; // 옵저버 식별자
};

struct MovingAverageObserverCustomOp : Ort::CustomOpBase<MovingAverageObserverCustomOp, MovingAverageObserverKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
    const char* GetName() const;
    const char* GetExecutionProviderType() const;
    size_t GetInputTypeCount() const;
    ONNXTensorElementDataType GetInputType(size_t index) const;
    size_t GetOutputTypeCount() const;
    ONNXTensorElementDataType GetOutputType(size_t index) const;
};

} // namespace MyQuantLib