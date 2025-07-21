# ONNX Runtime을 자동으로 다운로드하여 설정하는 헬퍼 모듈 (AIMET 방식)

# 1. CMake의 FetchContent 모듈 포함
include(FetchContent)

# 2. 설치된 파이썬 패키지로부터 ONNX Runtime 버전 가져오기
execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c" "import onnxruntime; print(onnxruntime.__version__)"
    RESULT_VARIABLE ORT_PYTHON_SEARCH_RESULT
    OUTPUT_VARIABLE ONNXRUNTIME_VERSION
    ERROR_VARIABLE ORT_PYTHON_SEARCH_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT ORT_PYTHON_SEARCH_RESULT EQUAL "0")
    message(FATAL_ERROR "Failed to get version from 'onnxruntime' package using Python interpreter '${Python3_EXECUTABLE}'. "
                        "Please ensure 'onnxruntime-gpu' is installed.\n"
                        "Error from Python: ${ORT_PYTHON_SEARCH_ERROR}")
endif()
message(STATUS "Found onnxruntime-gpu python package version: ${ONNXRUNTIME_VERSION}")

# 3. 현재 시스템에 맞는 다운로드 URL 구성
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set(PLATFORM_TAG "linux-x64")
    set(EXTENSION "tgz")
else()
    # 필요한 경우 Windows, macOS 등 다른 플랫폼에 대한 지원 추가
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# CUDA 빌드가 활성화되었으므로 "-gpu" 태그 추가
set(PLATFORM_TAG "${PLATFORM_TAG}-gpu")

# 최종 다운로드 URL 생성
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${PLATFORM_TAG}-${ONNXRUNTIME_VERSION}.${EXTENSION}")
message(NOTICE "ONNX Runtime: Fetching C++ distribution from ${ONNXRUNTIME_URL}")

# 4. FetchContent를 사용하여 ONNX Runtime 배포판 다운로드 및 압축 해제
FetchContent_Declare(
    onnxruntime_dist # FetchContent 항목의 이름
    URL ${ONNXRUNTIME_URL}
)
FetchContent_MakeAvailable(onnxruntime_dist)

# 5. FetchContent가 다운로드한 경로를 기반으로 변수 설정
# onnxruntime_dist_SOURCE_DIR 변수는 FetchContent가 자동으로 생성
set(ONNXRUNTIME_DOWNLOAD_DIR ${onnxruntime_dist_SOURCE_DIR})

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_DOWNLOAD_DIR}/include
    REQUIRED
)

find_library(ONNXRUNTIME_LIBRARIES onnxruntime
    HINTS ${ONNXRUNTIME_DOWNLOAD_DIR}/lib
    REQUIRED
)

message(STATUS "Found ONNXRuntime Headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNXRuntime Library: ${ONNXRUNTIME_LIBRARIES}")

# PARENT_SCOPE를 사용하여 이 변수들을 최상위 CMakeLists.txt에서 사용할 수 있게 함
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR} PARENT_SCOPE)
set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARIES} PARENT_SCOPE)