import onnx
import onnxruntime as ort
import numpy as np
import os
import custom_observer_ops # 빌드된 C++ 파이썬 모듈 임포트

def create_test_model(observer_name: str, output_path: str):
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, 3])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None, 3])
    
    observer_node = onnx.helper.make_node(
        'MovingAvgMinMaxObserver', inputs=['X'], outputs=['Y'],
        name='my_observer_node', domain='ai.my_ops',
        name=observer_name # C++ 커널에 전달될 속성
    )
    
    graph = onnx.helper.make_graph([observer_node], 'observer-graph', [X], [Y])
    model = onnx.helper.make_model(graph, producer_name='test-producer')
    onnx.save(model, output_path)
    print(f"Test model saved to {output_path}")

def run_test():
    print("--- Starting Custom Observer Op Test (CUDA) ---")
    
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available. Please install onnxruntime-gpu.")

    lib_path = os.path.abspath("../custom_observer_ops.so")
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Custom op library not found at {lib_path}")
    print(f"Loading custom op library from: {lib_path}")
    
    options = ort.SessionOptions()
    options.register_custom_ops_library(lib_path)
    
    model_path = "test_observer_model.onnx"
    observer_cpp_name = "activation_observer_1"
    create_test_model(observer_cpp_name, model_path)
    
    sess = ort.InferenceSession(model_path, options, providers=['CUDAExecutionProvider'])
    
    print("\n--- Running Inference and Observing Stats on CUDA ---")
    inputs = [
        np.array([[1.0, 2.0, 10.0]], dtype=np.float32),
        np.array([[-5.0, 0.0, 5.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    ]
    
    for i, input_data in enumerate(inputs):
        sess.run(None, {'X': input_data})
        stats = custom_observer_ops.get_stats(observer_cpp_name)
        print(f"Run {i+1}: input_min={input_data.min()}, input_max={input_data.max()}")
        print(f"  -> Observed stats from C++: {stats}\n")

    final_stats = custom_observer_ops.get_stats(observer_cpp_name)
    print(f"Final stats: min={final_stats.min:.4f}, max={final_stats.max:.4f}")
    assert abs(final_stats.min - 0.36) < 1e-5
    assert abs(final_stats.max - 8.75) < 1e-5
    print("\n--- CUDA Test Passed Successfully! ---")
    os.remove(model_path)

if __name__ == "__main__":
    run_test()
