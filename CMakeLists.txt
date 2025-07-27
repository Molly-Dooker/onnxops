# CMake 최소 버전 설정
cmake_minimum_required(VERSION 3.19)

# 프로젝트 이름 및 언어 설정
project(MyQuantLib LANGUAGES CXX CUDA)

# 모듈 경로 추가
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# CMAKE_SOURCE_DIR : 최상단 위치
# CMAKE_CURRENT_SOURCE_DIR : 현재 위치
# CMAKE_CURRENT_LIST_DIR : 현재 cmakelist 읽는 위치


# 의존성 처리 모듈 포함
include(PreparePyBind11)
include(PrepareONNX)

# --- 프로젝트 전역 설정 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- 빌드 옵션 ---
option(ENABLE_CUDA "Enable use of CUDA" ON)
option(BUILD_TESTS "Enable building tests" ON)

message(STATUS "MyQuantLib build configuration:")
message(STATUS "** ENABLE_CUDA = ${ENABLE_CUDA}")
message(STATUS "** BUILD_TESTS = ${BUILD_TESTS}")

# --- Python 의존성 찾기 ---
find_package(Python3 3.8 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python3: ${Python3_LIBRARIES}")

# --- 핵심 의존성 설정 ---
# PyBind11 (pip 설치된 버전 사용)
setup_pybind11()

# ONNX Runtime (환경 변수 또는 FetchContent 사용)
setup_onnxruntime()

# --- CUDA 설정 ---
if(ENABLE_CUDA)
    # 'cudart' 컴포넌트를 명시적으로 요청하여 CUDAToolkit::cudart 타겟 생성을 보장합니다.
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart)
    message(STATUS "Found CUDA Toolkit version ${CUDAToolkit_VERSION}")
    # 대상 GPU 아키텍처 설정 (필요에 따라 수정)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
else()
    message(WARNING "CUDA is disabled. The custom operator will not be built.")
endif()

# --- 서브디렉토리 빌드 ---
add_subdirectory(src)
add_subdirectory(python)

# --- 테스트 설정 ---
if(BUILD_TESTS)
    enable_testing()
    add_test(
        NAME python_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
endif()

message(STATUS "Project configuration finished.")