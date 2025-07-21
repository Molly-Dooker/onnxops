cmake_minimum_required(VERSION 3.19)
project(CustomObserver LANGUAGES CXX CUDA)

# --- 기본 설정 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# --- 모듈 경로 추가 ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- 필수 라이브러리 찾기 및 설정 ---
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(CUDA REQUIRED)

# 헬퍼 스크립트를 사용하여 ONNX Runtime과 Pybind11 설정
include(PrepareONNX)
include(PreparePyBind11)

# --- 인클루드 경로 설정 ---
include_directories(
    src
    ${Python3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${PYBIND11_INCLUDE_DIR}
)

# --- 소스 파일 정의 ---
set(SOURCES
    src/custom_op_onnx.cpp
    src/moving_avg_min_max.cpp
    src/python_bindings.cpp
    src/min_max_kernel.cu
)

# --- 파이썬 모듈 빌드 ---
add_library_pybind11(
    custom_observer_ops
    ${SOURCES}
)

# --- 라이브러리 링크 ---
target_link_libraries(custom_observer_ops PRIVATE
    ${ONNXRUNTIME_LIBRARIES}
    CUDA::cudart
)

# --- 빌드 후 출력 위치 지정 ---
set_target_properties(custom_observer_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../"
)