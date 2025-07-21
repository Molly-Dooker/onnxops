#include "observer.h"
#include "observer_kernel.cuh"
#include "state_manager.h"

namespace MyQuantLib {

MovingAverageObserverKernel::MovingAverageObserverKernel(const OrtApi&, const OrtKernelInfo* info) {
    Ort::ConstKernelInfo kernel_info(info);
    momentum_ = kernel_info.GetAttribute<float>("momentum");
    id_ = kernel_info.GetAttribute<std::string>("id");
}

void MovingAverageObserverKernel::Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    // [수정] auto를 사용하여 정확한 타입(const)을 추론하도록 함
    auto input_tensor = ctx.GetInput(0);
    const float* X_data = input_tensor.GetTensorData<float>();

    // [수정] auto를 사용하여 정확한 타입(const)을 추론하도록 함
    auto X_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto& shape = X_info.GetShape();
    int64_t num_elements = X_info.GetElementCount();

    // [수정] auto를 사용하여 rvalue를 올바르게 받도록 함
    auto output_tensor = ctx.GetOutput(0, shape.data(), shape.size());
    float* Y_data = output_tensor.GetMutableTensorData<float>();

    ObserverState* state = StateManager::get_instance().get_state_ptr(id_);

    cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();
    if (!stream) {
        throw std::runtime_error("Failed to get CUDA stream from context.");
    }

    launch_observer_kernel(
        X_data, Y_data, num_elements,
        state,
        momentum_, stream
    );
}

void* MovingAverageObserverCustomOp::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new MovingAverageObserverKernel(api, info);
}

const char* MovingAverageObserverCustomOp::GetName() const {
    return "MovingAverageObserver";
}

const char* MovingAverageObserverCustomOp::GetExecutionProviderType() const {
    return "CUDAExecutionProvider";
}

size_t MovingAverageObserverCustomOp::GetInputTypeCount() const {
    return 1;
}

ONNXTensorElementDataType MovingAverageObserverCustomOp::GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t MovingAverageObserverCustomOp::GetOutputTypeCount() const {
    return 1;
}

ONNXTensorElementDataType MovingAverageObserverCustomOp::GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

} // namespace MyQuantLib