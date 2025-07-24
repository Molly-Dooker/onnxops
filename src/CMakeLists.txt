add_library(my_quant_ops SHARED)

target_sources(my_quant_ops
    PRIVATE
      histogram_kernel.cu
      observer_kernel.cu
      state_manager.cpp
      observer.cu              # 바뀐 확장자
      custom_op_library.cpp
)

# separable compilation 켜기
set_target_properties(my_quant_ops PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(my_quant_ops
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
    $<$<BOOL:${ENABLE_CUDA}>:${CUDAToolkit_INCLUDE_DIRS}>
)

target_link_libraries(my_quant_ops
    PUBLIC
      ONNXRuntime::onnxruntime
    PRIVATE
      $<$<BOOL:${ENABLE_CUDA}>:CUDA::cudart>
      $<$<BOOL:${ENABLE_CUDA}>:CUDA::cuda_driver>
)

set_target_properties(my_quant_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)