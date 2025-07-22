import os
import onnx
import onnxruntime as ort
import numpy as np
import my_quant_lib  # 수정된 lib를 빌드/설치하여 import
from numpy.random import default_rng
import ipdb
# 테스트용 ONNX 모델 생성 함수
def create_test_model(observer_id, momentum=0.9, dimension=4 ,model_path="observer_test.onnx"):
    # 입력/출력 텐서 정의 (float32, 2D 텐서, 가변 크기)
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None]*dimension)
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None]*dimension)
    # MovingAverageObserver 노드 생성
    observer_node = onnx.helper.make_node(
        'MovingAverageObserver',
        inputs=['X'],
        outputs=['Y'],
        domain='com.my-quant-lib',   # Custom Op 도메인
        id=observer_id,
        momentum=float(momentum)
    )
    # 그래프와 모델 작성
    graph = onnx.helper.make_graph([observer_node], 'test-observer-graph', [X], [Y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 15),                   # ONNX 기본 도메인 opset 15
            onnx.helper.make_opsetid("com.my-quant-lib", 1)     # 커스텀 도메인 opset 1
        ],
        ir_version=10
    )
    # model.ir_version = 10
    onnx.save(model, model_path)
    return model_path

# 1. 테스트 모델 생성 (momentum=0.9, ID="obs1")
model_path = create_test_model("obs1", momentum=0.9)

# 2. StateManager에 해당 observer ID 등록 (초기 상태 세팅)
my_quant_lib.register_observer("obs1")

# 3. ONNX Runtime 세션 생성 (커스텀 Op 라이브러리 로드 및 CUDA EP 지정)
so = ort.SessionOptions()
# 빌드된 커스텀 연산자 공유 라이브러리(.so) 경로 설정
lib_path = os.path.join(os.path.dirname(my_quant_lib.__file__), "libmy_quant_ops.so")
so.register_custom_ops_library(lib_path)
# CUDA Execution Provider 사용하여 세션 생성
# ipdb.set_trace()
session = ort.InferenceSession(model_path, so, providers=['CUDAExecutionProvider'])

# 4. 여러 번에 걸쳐 입력 데이터를 흘려보내며 상태(min/max) 추적
print("Iter | Input_Min    Input_Max    State_Min    State_Max")
min_= None; max_= None;
for i in range(100):
    # 난수 입력 생성 (점차 값의 범위를 변화시켜 상태 변화를 관찰)
    data = (default_rng().random((512,3,128,128), dtype=np.float32) * (10 - i)) + i  # iteration에 따라 분포 변경
    # data =np.random.rand(512,1204).astype(np.float32)
    out = session.run(None, {'X': data})[0]
    # 출력이 입력과 같은지 검증
    assert np.array_equal(out, data), f"Output differs from input at iter {i}"
    # 현재 상태 가져오기
    state = my_quant_lib.get_observer_state("obs1")
    # 입력과 상태의 min/max 출력
    if min_ is None:
        min_ = data.min(); max_ = data.max()
    else:
        min_ = min_*0.9+ data.min()*0.1; max_ = max_*0.9+ data.max()*0.1; 
    
    print(f'max:{max_:4.10f}, {state.max:4.10f}')
    print(f'min:{min_:4.10f}, {state.min:4.10f}')
    print("-------------------------------------------------")
