# 커스텀 연산자 라이브러리 생성
add_library(my_quant_ops SHARED)

# 소스 파일 지정
target_sources(my_quant_ops
    PRIVATE
        histogram_kernel.cu
        state_manager.cpp
        observer.cpp
        observer_kernel.cu
        custom_op_library.cpp
)

# 헤더 파일 경로 포함
target_include_directories(my_quant_ops
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# [수정된 부분] ONNXRuntime::onnxruntime을 PUBLIC으로 링크하여
# 헤더 경로와 라이브러리 정보를 이 타겟을 링크하는 다른 타겟에게 전파합니다.
target_link_libraries(my_quant_ops
    PUBLIC
        ONNXRuntime::onnxruntime
)

# 출력 디렉토리 설정 (Python에서 찾기 쉽도록)
set_target_properties(my_quant_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)