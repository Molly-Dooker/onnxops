# Python 모듈 생성
add_library(my_quant_lib_python MODULE bindings.cpp)
install(TARGETS my_quant_lib_python LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/python/my_quant_lib")
install(TARGETS my_quant_ops        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/python/my_quant_lib")

# 타겟 이름 변경 (Python 모듈 이름과 동일하게)
set_target_properties(my_quant_lib_python PROPERTIES
    PREFIX ""
    OUTPUT_NAME "my_quant_lib"
)

# 라이브러리 링크
target_link_libraries(my_quant_lib_python
    PRIVATE
        my_quant_ops
        pybind11::pybind11
)

# 출력 디렉토리 설정
set_target_properties(my_quant_lib_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/my_quant_lib"
)