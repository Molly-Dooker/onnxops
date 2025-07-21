# ONNX Runtime을 찾고 관련 변수를 설정하는 헬퍼 모듈
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os; import onnxruntime; print(os.path.dirname(onnxruntime.__file__))"
    OUTPUT_VARIABLE ONNXRUNTIME_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(ONNXRUNTIME_PYTHON_PATH AND EXISTS "${ONNXRUNTIME_PYTHON_PATH}/capi/onnxruntime_c_api.h")
    set(ORT_CAPI_PATH "${ONNXRUNTIME_PYTHON_PATH}/capi")
    set(ORT_LIB_PATH "${ONNXRUNTIME_PYTHON_PATH}/lib")
else()
    message(FATAL_ERROR "Could not find onnxruntime C API headers. Please install 'onnxruntime-gpu'.")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS ${ORT_CAPI_PATH} REQUIRED)
find_library(ONNXRUNTIME_LIBRARIES onnxruntime HINTS ${ORT_LIB_PATH} REQUIRED)

message(STATUS "Found ONNXRuntime Headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNXRuntime Library: ${ONNXRUNTIME_LIBRARIES}")

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR} PARENT_SCOPE)
set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARIES} PARENT_SCOPE)