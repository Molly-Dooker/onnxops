# 이 모듈은 ONNX Runtime을 찾거나 다운로드하는 함수를 제공합니다.

include(FetchContent)

function(setup_onnxruntime)
    if (TARGET ONNXRuntime::onnxruntime)
        return()
    endif()

    # 1. 사용자가 제공한 경로를 먼저 시도
    # ONNXRUNTIME_ROOT 환경변수나 CMake 변수를 통해 찾을 수 있도록 find_package를 사용
    find_package(ONNXRuntime QUIET)

    if (ONNXRuntime_FOUND)
        message(STATUS "Found ONNX Runtime via find_package (e.g., ONNXRUNTIME_ROOT).")
        # find_package가 IMPORTED 타겟을 생성했다고 가정
        # 만약 타겟이 없다면 여기서 생성
        if (NOT TARGET ONNXRuntime::onnxruntime)
             add_library(ONNXRuntime::onnxruntime SHARED IMPORTED)
             set_target_properties(ONNXRuntime::onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
            )
        endif()
        return()
    endif()

    # 2. 찾지 못했다면 FetchContent로 다운로드
    message(NOTICE "ONNX Runtime not found locally. Fetching from GitHub...")

    execute_process(
        COMMAND ${Python3_EXECUTABLE} "-c" "import onnxruntime; print(onnxruntime.__version__)"
        OUTPUT_VARIABLE ONNXRUNTIME_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (NOT ONNXRUNTIME_VERSION)
        message(FATAL_ERROR "Failed to get onnxruntime version. Please install with 'pip install onnxruntime-gpu'")
    endif()
    message(STATUS "Detected onnxruntime version: ${ONNXRUNTIME_VERSION}")

    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        set(PLATFORM_TAG "linux-x64")
        set(EXTENSION "tgz")
    elseif ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
        set(PLATFORM_TAG "win-x64")
        set(EXTENSION "zip")
    elseif ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
        set(PLATFORM_TAG "osx-arm64") # Apple Silicon 가정이지만, x86_64일 수도 있음
        set(EXTENSION "tgz")
    else()
        message(FATAL_ERROR "Unsupported system: ${CMAKE_SYSTEM_NAME}")
    endif()

    if (ENABLE_CUDA)
        set(PLATFORM_TAG "${PLATFORM_TAG}-gpu")
    endif()

    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${PLATFORM_TAG}-${ONNXRUNTIME_VERSION}.${EXTENSION}")
    message(STATUS "Downloading from: ${ONNXRUNTIME_URL}")

    FetchContent_Declare(
        onnxruntime_dep
        URL ${ONNXRUNTIME_URL}
    )
    FetchContent_MakeAvailable(onnxruntime_dep)

    set(ORT_INCLUDE_DIR ${onnxruntime_dep_SOURCE_DIR}/include)
    set(ORT_LIB_DIR ${onnxruntime_dep_SOURCE_DIR}/lib)
    find_library(ORT_LIBRARY onnxruntime NAMES onnxruntime libonnxruntime.so HINTS ${ORT_LIB_DIR})

    add_library(ONNXRuntime::onnxruntime SHARED IMPORTED)
    set_target_properties(ONNXRuntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
    )
endfunction()