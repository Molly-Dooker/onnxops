# 이 모듈은 pip로 설치된 pybind11을 찾는 매크로를 제공합니다.

macro(setup_pybind11)
    if (NOT TARGET pybind11::pybind11)
        execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import pybind11; print(pybind11.get_include())"
            RESULT_VARIABLE PYBIND11_RESULT
            OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(PYBIND11_RESULT EQUAL 0)
            message(STATUS "Found pybind11 include path: ${PYBIND11_INCLUDE_DIR}")
        else()
            message(FATAL_ERROR "pybind11 not found. Please install it using 'pip install pybind11'")
        endif()

        add_library(pybind11::pybind11 INTERFACE IMPORTED)
        target_include_directories(pybind11::pybind11 INTERFACE ${PYBIND11_INCLUDE_DIR})
        target_link_libraries(pybind11::pybind11 INTERFACE Python3::Module)
    endif()
endmacro()