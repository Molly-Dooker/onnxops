import os
import sys
import onnx
import onnxruntime as ort
import numpy as np
import ipdb
# # (1) 빌드된 파이썬 모듈 경로 추가
# #    -- CMake 출력 디렉토리 기준으로 경로를 맞춰주세요.
# module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                           "..", "build", "python", "my_quant_lib"))
# sys.path.insert(0, module_dir)

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
        opset_imports=[onnx.helper.make_opsetid("", 15)]
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
    lib_path = my_quant_lib.get_library_path()
    so.register_custom_ops_library(lib_path)

    # Providers 설정
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    sess = ort.InferenceSession(MODEL_PATH, so, providers=providers)

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
