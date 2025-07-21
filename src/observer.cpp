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

    const OrtValue* X = ctx.GetInput(0);
    const float* X_data = X->GetTensorData<float>();

    Ort::TensorTypeAndShapeInfo X_info = X->GetTensorTypeAndShapeInfo();
    auto& shape = X_info.GetShape();
    OrtValue* Y = ctx.GetOutput(0, shape.data(), shape.size());
    float* Y_data = Y->GetMutableTensorData<float>();
    
    int64_t num_elements = X_info.GetElementCount();

    ObserverState* state = StateManager::get_instance().get_state_ptr(id_);

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    Ort::Custom::CudaContext cuda_ctx;
    cuda_ctx.Init(*api, context, nullptr);
    cudaStream_t stream = cuda_ctx.GetStream();

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