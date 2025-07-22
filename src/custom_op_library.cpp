#include "observer.h"
#include "onnxruntime_c_api.h"

static MyQuantLib::MovingAverageObserverOp_CPU g_op_cpu;
static MyQuantLib::MovingAverageObserverOp_CUDA g_op_cuda;
static OrtCustomOpDomain* g_domain = nullptr;

extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  const OrtApi* ort = api_base->GetApi(ORT_API_VERSION);
  if (!g_domain) {
    ort->CreateCustomOpDomain("com.my-quant-lib", &g_domain);
    // CPU, CUDA 모두 등록
    ort->CustomOpDomain_Add(g_domain, &g_op_cpu);
    ort->CustomOpDomain_Add(g_domain, &g_op_cuda);
  }
  return ort->AddCustomOpDomain(options, g_domain);
}
