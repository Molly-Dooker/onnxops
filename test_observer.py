import os
import platform
import onnx
import onnxruntime as ort
import numpy as np
import ipdb
import my_quant_lib

CUSTOM_OP_DOMAIN = 'com.my-quant-lib'
MODEL_PATH = "test_observer_model_stateful.onnx"

def create_test_model(observer_id, momentum=0.9):
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, None])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None, None])
    observer_node = onnx.helper.make_node(
        'MovingAverageObserver',
        inputs=['X'],
        outputs=['Y'],
        domain=CUSTOM_OP_DOMAIN,
        id=observer_id,
        momentum=float(momentum)
    )
    graph = onnx.helper.make_graph(
        [observer_node],
        'observer-graph-stateful',
        [X],
        [Y]
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 15),
                       onnx.helper.make_opsetid(CUSTOM_OP_DOMAIN, 1)],
        ir_version=10
    )
    onnx.save(model, MODEL_PATH)

def main():
    obs_id = "test_obs_1"
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    
    create_test_model(obs_id)
    # Observer 등록
    my_quant_lib.register_observer(obs_id)

    # SessionOptions + custom op 라이브러리 로드
    so = ort.SessionOptions()
    pkg_dir = os.path.dirname(my_quant_lib.__file__)
    lib_name = {
        "Windows": "my_quant_ops.dll",
        "Darwin":  "libmy_quant_ops.dylib"
    }.get(platform.system(), "libmy_quant_ops.so")
    lib_path = os.path.join(pkg_dir, lib_name)
    so.register_custom_ops_library(lib_path)

    # Providers 설정
    providers = ['CUDAExecutionProvider']
    ipdb.set_trace()
    sess = ort.InferenceSession(MODEL_PATH, so)
    sess.set_providers(providers)

    # 10회 inference 및 state 추적
    for i in range(10):
        data = (np.random.rand(10, 20).astype(np.float32) * (10 - i)) + (i / 2.0)
        outputs = sess.run(None, {'X': data})[0]
        assert np.array_equal(data, outputs), f"Output mismatch at iter {i}"
        state = my_quant_lib.get_observer_state(obs_id)
        print(f"[Iter {i+1}] min={state.min:.4f}, max={state.max:.4f}")

    final = my_quant_lib.get_observer_state(obs_id)
    print(f"Final state: min={final.min:.4f}, max={final.max:.4f}")

    assert 4.0 < final.min < 5.0, "Final min out of expected range"
    assert 5.0 < final.max < 6.0, "Final max out of expected range"
    print("All tests passed!")

if __name__ == "__main__":
    main()
