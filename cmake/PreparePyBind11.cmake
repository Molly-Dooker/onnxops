# Pybind11을 찾고 관련 헬퍼 매크로를 정의하는 모듈 (AIMET 방식 적용)

# 1. Python 인터프리터를 직접 실행하여 설치된 pybind11 패키지의 include 경로를 찾음
execute_process(
    COMMAND ${Python3_EXECUTABLE} "-c" "import pybind11; print(pybind11.get_include())"
    RESULT_VARIABLE PYBIND11_SEARCH_RESULT
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    ERROR_VARIABLE PYBIND11_SEARCH_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 2. pybind11를 찾는 데 실패하면, 에러 메시지와 함께 빌드를 중단
if(NOT PYBIND11_SEARCH_RESULT EQUAL "0")
    message(FATAL_ERROR "Failed to find 'pybind11' include path using Python interpreter '${Python3_EXECUTABLE}'. "
                        "Please run 'pip install pybind11'.\n"
                        "Error from Python: ${PYBIND11_SEARCH_ERROR}")
endif()

# 3. 찾은 경로를 PARENT_SCOPE를 통해 최상위 CMakeLists.txt에서도 사용할 수 있도록 설정
set(PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDE_DIR} PARENT_SCOPE)
message(STATUS "Found PyBind11 include directory: ${PYBIND11_INCLUDE_DIR}")


# 4. C++ 소스를 파이썬 모듈로 빌드하는 헬퍼 매크로 정의
# FetchContent를 사용하지 않으므로, pybind11_add_module() 함수를 직접 사용할 수 없음.
# 따라서 AIMET처럼 라이브러리를 수동으로 정의하는 매크로를 만듦.
macro(add_library_pybind11 target_name)
    # add_library를 사용하여 공유 라이브러리(파이썬 모듈) 생성
    add_library(
        ${target_name}
        SHARED
        ${ARGN} # 매크로에 전달된 모든 추가 인자 (소스 파일 목록)
    )

    # 생성된 라이브러리에 속성 설정
    target_link_libraries(${target_name} PRIVATE
        ${Python3_LIBRARIES}
    )

    # Pybind11 헤더를 찾을 수 있도록 인클루드 경로 추가
    target_include_directories(${target_name} PRIVATE
        ${PYBIND11_INCLUDE_DIR}
    )

    # 파이썬 모듈로서 올바른 파일 확장자(.so, .pyd 등)와 접두사가 없도록 설정
    set_target_properties(${target_name} PROPERTIES
        PREFIX ""
        SUFFIX ".so" # Linux 기준. CMake가 Windows(.pyd) 등 플랫폼에 맞게 조정함.
    )
endmacro()