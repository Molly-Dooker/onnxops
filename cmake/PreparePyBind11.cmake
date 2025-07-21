# Pybind11을 찾고 관련 헬퍼 매크로를 정의하는 모듈
include(FetchContent)

# pybind11가 시스템에 없는 경우를 대비하여 FetchContent 선언
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.0 # 특정 버전 고정
)

# find_package를 먼저 시도하고, 실패하면 FetchContent를 사용
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package. Fetching from source...")
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "Found pybind11 via find_package.")
endif()

# PARENT_SCOPE를 사용하여 pybind11_INCLUDE_DIRS를 전역적으로 사용 가능하게 함
set(PYBIND11_INCLUDE_DIR ${pybind11_INCLUDE_DIRS} PARENT_SCOPE)
message(STATUS "Set PyBind11 include directory: ${PYBIND11_INCLUDE_DIR}")


# C++ 소스를 파이썬 모듈로 빌드하는 헬퍼 매크로 정의
macro(add_library_pybind11 target_name)
    # pybind11_add_module는 pybind11을 add_subdirectory 또는 FetchContent로 포함하면 사용 가능
    pybind11_add_module(
        ${target_name}
        SHARED
        ${ARGN} # 매크로에 전달된 모든 추가 인자 (소스 파일 목록)
    )
endmacro()