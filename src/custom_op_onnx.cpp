#include "onnxruntime_cxx_api.h"
#include "moving_avg_min_max.h"
#include "min_max_kernel.h" // CUDA 커널 헤더 포함
#include <vector>
#include <cuda_runtime.h>

struct MovingAvgMinMaxObserverKernel {
    MovingAvgMinMaxObserverKernel(const OrtApi& api, const OrtKernelInfo* info) {
        Ort::CustomOpApi custom_api(api);
        name_ = custom_api.KernelInfoGetAttribute<std::string>(info, "name");

        // CUDA 결과(min/max)를 저장할 GPU 메모리 사전 할당
        cudaMalloc(&result_gpu_, 2 * sizeof(float));
    }

    ~MovingAvgMinMaxObserverKernel() {
        cudaFree(result_gpu_);
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);

        // 1. GPU에 있는 입력 텐서 가져오기
        const OrtValue* input_tensor = ctx.GetInput(0);
        const float* input_data_gpu = input_tensor->GetTensorData<float>();
        OrtTensorDimensions dimensions(input_tensor);
        size_t element_count = dimensions.Size();

        // 2. CUDA 커널을 호출하여 GPU에서 min/max 계산
        launch_min_max_kernel(input_data_gpu, element_count, result_gpu_);

        // 3. GPU에서 계산된 결과를 CPU로 복사
        float result_cpu[2]; // min, max
        cudaMemcpy(result_cpu, result_gpu_, 2 * sizeof(float), cudaMemcpyDeviceToHost);
        float min_val = result_cpu[0];
        float max_val = result_cpu[1];

        // 4. ObserverManager를 통해 통계 업데이트 (CPU에서 수행)
        ObserverManager::update_stats(name_, min_val, max_val);

        // 5. 출력을 입력과 동일하게 설정 (GPU 메모리 복사)
        OrtValue* output_tensor = ctx.GetOutput(0, dimensions.GetDims());
        float* output_data_gpu = output_tensor->GetTensorMutableData<float>();
        cudaMemcpy(output_data_gpu, input_data_gpu, element_count * sizeof(float), cudaMemcpyDeviceToDevice);
    }

private:
    std::string name_;
    float* result_gpu_ = nullptr; // min/max 결과를 저장할 GPU 버퍼
};

// 커스텀 연산자 정의 (실행 프로바이더를 CUDA로 변경)
struct MovingAvgMinMaxObserverCustomOp : Ort::CustomOpBase<MovingAvgMinMaxObserverCustomOp, MovingAvgMinMaxObserverKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new MovingAvgMinMaxObserverKernel(api, info);
    }
    const char* GetName() const { return "MovingAvgMinMaxObserver"; }
    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; } // <-- 중요
    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

    void GetAttributes(Ort::CustomOpApi& api) {
        api.AddRequiredAttribute("name", Ort::AttributeType::STRING);
    }
};

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    Ort::Global<void>::api_ = api_base->GetApi(ORT_API_VERSION);
    Ort::CustomOpDomain domain("ai.my_ops");
    static MovingAvgMinMaxObserverCustomOp op; // 정적 객체로 변경
    domain.Add(&op);
    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    return nullptr;
}