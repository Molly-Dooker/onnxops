// src/custom_op_library.cpp

#include "observer.h"
#include "onnxruntime_c_api.h"  // C API 헤더

// 1) 전역 custom-op 인스턴스
static MyQuantLib::MovingAverageObserverCustomOp g_MovingAverageObserver;

// 2) 전역 도메인 포인터 (한 번만 생성)
static OrtCustomOpDomain* g_custom_op_domain = nullptr;

// 3) RegisterCustomOps 심볼(export) 정의
extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  // C API 가져오기
  const OrtApi* ort = api_base->GetApi(ORT_API_VERSION);

  // 도메인이 아직 생성되지 않았다면 한 번만 생성 & op 등록
  if (g_custom_op_domain == nullptr) {
    ort->CreateCustomOpDomain("com.my-quant-lib", &g_custom_op_domain);
    ort->CustomOpDomain_Add(g_custom_op_domain, &g_MovingAverageObserver);
  }

  // SessionOptions에 도메인 붙이기
  return ort->AddCustomOpDomain(options, g_custom_op_domain);
}
