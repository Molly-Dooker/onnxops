import onnx
import onnxruntime as ort
import numpy as np
import my_quant_lib
import pytest
import os

CUSTOM_OP_DOMAIN = 'com.my-quant-lib'
MODEL_PATH = "test_observer_model_stateful.onnx"

def create_test_model(observer_id, momentum=0.9):
    """상태 저장소와 연동되는 MovingAverageObserver 모델을 생성합니다."""
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, None])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None, None])

    observer_node = onnx.helper.make_node(
        'MovingAverageObserver',
        inputs=['X'],
        outputs=,
        domain=CUSTOM_OP_DOMAIN,
        id=observer_id,
        momentum=float(momentum)
    )

    graph = onnx.helper.make_graph(
        [observer_node],
        'observer-graph-stateful',
        [X],
       
    )

    opset_imports = [onnx.helper.make_opsetid("", 15)]
    model = onnx.helper.make_model(graph, opset_imports=opset_imports)
    onnx.save(model, MODEL_PATH)
    return MODEL_PATH

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """테스트 모델을 생성하고 테스트 후 정리합니다."""
    observer_id = "test_obs_1"
    create_test_model(observer_id)
    yield
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

def test_stateful_moving_average_observer():
    observer_id = "test_obs_1"

    my_quant_lib.register_observer(observer_id)

    so = ort.SessionOptions()
    lib_path = my_quant_lib.get_library_path()
    so.register_custom_ops_library(lib_path)

    providers =
    sess = ort.InferenceSession(MODEL_PATH, so, providers=providers)

    num_iterations = 10
    for i in range(num_iterations):
        input_data = (np.random.rand(10, 20).astype(np.float32) * (10 - i)) + (i / 2.0)
        outputs = sess.run(None, {'X': input_data})
        
        np.testing.assert_array_equal(input_data, outputs)

        state = my_quant_lib.get_observer_state(observer_id)
        print(f"Iteration {i+1}: Min={state.min:.4f}, Max={state.max:.4f}")

    final_state = my_quant_lib.get_observer_state(observer_id)
    print(f"\nFinal Result: Min={final_state.min:.4f}, Max={final_state.max:.4f}")

    assert 4.0 < final_state.min < 5.0
    assert 5.0 < final_state.max < 6.0
    print("Test passed!")