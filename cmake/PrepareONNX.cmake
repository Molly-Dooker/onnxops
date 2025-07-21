# ONNX Runtime을 찾고 관련 변수를 설정하는 헬퍼 모듈

# 1. onnxruntime 파이썬 패키지의 위치를 찾음
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, onnxruntime; print(os.path.dirname(onnxruntime.__file__))"
    RESULT_VARIABLE ORT_PYTHON_SEARCH_RESULT
    OUTPUT_VARIABLE ONNXRUNTIME_PYTHON_PATH
    ERROR_VARIABLE ORT_PYTHON_SEARCH_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 2. 파이썬 명령이 성공했는지 확인
if(NOT ORT_PYTHON_SEARCH_RESULT EQUAL "0")
    message(FATAL_ERROR "Failed to find 'onnxruntime' package using Python interpreter '${Python3_EXECUTABLE}'. "
                        "Please ensure 'onnxruntime-gpu' is installed in this specific Python environment.\n"
                        "Error from Python: ${ORT_PYTHON_SEARCH_ERROR}")
endif()

# 3. 찾은 경로를 기반으로 C API 헤더 파일 경로를 구성
set(ORT_CAPI_H_PATH "${ONNXRUNTIME_PYTHON_PATH}/capi/onnxruntime_c_api.h")

# 4. 해당 경로에 헤더 파일이 실제로 존재하는지 확인
if(NOT EXISTS "${ORT_CAPI_H_PATH}")
    message(FATAL_ERROR "Could not find 'onnxruntime_c_api.h'.\n"
                        "Checked path: '${ORT_CAPI_H_PATH}'.\n"
                        "This might indicate a broken onnxruntime-gpu installation. "
                        "Please try reinstalling it.")
endif()

# 5. 최종 변수 설정
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_PYTHON_PATH}/capi")
find_library(ONNXRUNTIME_LIBRARIES onnxruntime HINTS "${ONNXRUNTIME_PYTHON_PATH}/lib" REQUIRED)

message(STATUS "Found ONNXRuntime Headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNXRuntime Library: ${ONNXRUNTIME_LIBRARIES}")

# PARENT_SCOPE를 사용하여 이 변수들을 최상위 CMakeLists.txt에서 사용할 수 있게 함
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR} PARENT_SCOPE)
set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARIES} PARENT_SCOPE)