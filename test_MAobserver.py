import os
import onnx
import onnxruntime as ort
import numpy as np
import my_quant_lib
from numpy.random import default_rng
from torch.ao.quantization import HistogramObserver
def create_test_model(observer_id, momentum=0.9, dimension=4 ,model_path="observer_test.onnx"):
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None]*dimension)
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None]*dimension)
    observer_node = onnx.helper.make_node(
        'MovingAverageObserver',
        inputs=['X'],
        outputs=['Y'],
        domain='com.my-quant-lib',   # Custom Op 도메인
        name=observer_id,
        momentum=float(momentum)
    )
    graph = onnx.helper.make_graph([observer_node], 'test-observer-graph', [X], [Y])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 15),                   # ONNX 기본 도메인 opset 15
            onnx.helper.make_opsetid("com.my-quant-lib", 1)     # 커스텀 도메인 opset 1
        ],
        ir_version=10
    )
    onnx.save(model, model_path)
    return model_path

model_path = create_test_model("obs1", momentum=0.9)
my_quant_lib.register_moving_average_observer("obs1")
so = ort.SessionOptions()
lib_path = os.path.join(os.path.dirname(my_quant_lib.__file__), "libmy_quant_ops.so")
so.register_custom_ops_library(lib_path)
session = ort.InferenceSession(model_path, so, providers=['CUDAExecutionProvider'])
print("Iter | Input_Min    Input_Max    State_Min    State_Max")
min_= None; max_= None;
for i in range(100):
    data = (default_rng().random((512,3,128,128), dtype=np.float32) * (10 - i)) + i
    out = session.run(None, {'X': data})[0]
    assert np.array_equal(out, data), f"Output differs from input at iter {i}"
    state = my_quant_lib.get_observer_state("obs1")
    if min_ is None:
        min_ = data.min(); max_ = data.max()
    else:
        min_ = min_*0.9+ data.min()*0.1; max_ = max_*0.9+ data.max()*0.1; 
    abs_err_max = abs(state.max - max_)
    abs_err_min = abs(state.min - min_)
    rel_err_max = abs_err_max / (abs(max_) + 1e-12) * 100  # %
    rel_err_min = abs_err_min / (abs(min_) + 1e-12) * 100  # %
    print(f"Iter {i:3d} | "
          f"Max Err: ({rel_err_max:.4f}%) | "
          f"Min Err: ({rel_err_min:.4f}%)")