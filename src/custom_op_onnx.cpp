#include "onnxruntime_cxx_api.h"
#include "moving_avg_min_max.h"
#include "min_max_kernel.h"
#include <cuda_runtime.h>

struct MovingAvgMinMaxObserverKernel {
    MovingAvgMinMaxObserverKernel(const OrtApi& api, const OrtKernelInfo* info) {
        Ort::CustomOpApi custom_api(api);
        name_ = custom_api.KernelInfoGetAttribute<std::string>(info, "name");
        cudaMalloc(&result_gpu_, 2 * sizeof(float));
    }

    ~MovingAvgMinMaxObserverKernel() { cudaFree(result_gpu_); }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        const OrtValue* input_tensor = ctx.GetInput(0);
        const float* input_data_gpu = input_tensor->GetTensorData<float>();
        OrtTensorDimensions dimensions(input_tensor);
        
        launch_min_max_kernel(input_data_gpu, dimensions.Size(), result_gpu_);

        float result_cpu[2];
        cudaMemcpy(result_cpu, result_gpu_, 2 * sizeof(float), cudaMemcpyDeviceToHost);
        
        ObserverManager::update_stats(name_, result_cpu[0], result_cpu[1]);

        OrtValue* output_tensor = ctx.GetOutput(0, dimensions.GetDims());
        float* output_data_gpu = output_tensor->GetTensorMutableData<float>();
        cudaMemcpy(output_data_gpu, input_data_gpu, dimensions.Size() * sizeof(float), cudaMemcpyDeviceToDevice);
    }
private:
    std::string name_;
    float* result_gpu_ = nullptr;
};

struct MovingAvgMinMaxObserverCustomOp : Ort::CustomOpBase<MovingAvgMinMaxObserverCustomOp, MovingAvgMinMaxObserverKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return new MovingAvgMinMaxObserverKernel(api, info); }
    const char* GetName() const { return "MovingAvgMinMaxObserver"; }
    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    void GetAttributes(Ort::CustomOpApi& api) { api.AddRequiredAttribute("name", Ort::AttributeType::STRING); }
};

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    Ort::Global<void>::api_ = api_base->GetApi(ORT_API_VERSION);
    Ort::CustomOpDomain domain("ai.my_ops");
    static MovingAvgMinMaxObserverCustomOp op;
    domain.Add(&op);
    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    return nullptr;
}
