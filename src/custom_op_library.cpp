#include "observer.h"
#include "onnxruntime_cxx_api.h"

static MyQuantLib::MovingAverageObserverCustomOp g_MovingAverageObserver;

#ifdef _WIN32
#define OP_EXPORT __declspec(dllexport)
#else
#define OP_EXPORT
#endif

extern "C" OP_EXPORT OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
    Ort::CustomOpDomain domain("com.my-quant-lib");
    domain.Add(&g_MovingAverageObserver);
    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    return nullptr;
}