# Pybind11을 찾고 관련 헬퍼 매크로를 정의하는 모듈
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.0
)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package. Fetching from source...")
    FetchContent_MakeAvailable(pybind11)
else()
    message(STATUS "Found pybind11 via find_package.")
endif()

set(PYBIND11_INCLUDE_DIR ${pybind11_INCLUDE_DIRS} PARENT_SCOPE)
message(STATUS "Set PyBind11 include directory: ${PYBIND11_INCLUDE_DIR}")

# C++ 소스를 파이썬 모듈로 빌드하는 헬퍼 매크로 정의
macro(add_library_pybind11 target_name)
    pybind11_add_module(
        ${target_name}
        SHARED
        ${ARGN}
    )
endmacro()