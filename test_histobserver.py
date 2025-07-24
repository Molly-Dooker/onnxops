# test_histogram_observer.py

import os
import onnx
import onnxruntime as ort
import numpy as np
import my_quant_lib
from numpy.random import default_rng
import numpy.testing as npt
import ipdb
def create_histogram_model(observer_id, bins, dimension=2, model_path="histogram_test.onnx"):
    """
    Creates an ONNX model with a single HistogramObserver node.
    - observer_id: string ID for the observer
    - bins: number of histogram bins
    - dimension: rank of the input tensor (default 2D)
    """
    # make shape [None, None, ...] of length `dimension`
    shape = [None] * dimension
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, shape)
    # H = onnx.helper.make_tensor_value_info('H', onnx.TensorProto.INT64, [bins])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, shape)

    node = onnx.helper.make_node(
        'HistogramObserver',
        inputs=['X'],
        outputs=['Y'],
        domain='com.my-quant-lib',
        id=observer_id,
        bins=bins
    )

    graph = onnx.helper.make_graph(
        [node],
        'histogram-observer-graph',
        [X],
        [Y]
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 15),
            onnx.helper.make_opsetid("com.my-quant-lib", 1)
        ],
        ir_version=10
    )
    onnx.save(model, model_path)
    return model_path

if __name__ == "__main__":
    OBS_ID = "hist1"
    BINS = 2048

    # 1. build and save the test model
    model_path = create_histogram_model(OBS_ID, dimension=4,bins=BINS)

    # 2. register the histogram observer in Python
    my_quant_lib.register_histogram_observer(OBS_ID, BINS)

    # 3. create ONNX Runtime session (CPU EP to ensure state update)
    so = ort.SessionOptions()
    lib_path = os.path.join(os.path.dirname(my_quant_lib.__file__), "libmy_quant_ops.so")
    so.register_custom_ops_library(lib_path)
    sess = ort.InferenceSession(model_path, so, providers=["CUDAExecutionProvider"])
    for i in range(100):
        # 난수 입력 생성 (점차 값의 범위를 변화시켜 상태 변화를 관찰)
        data = (default_rng().random((512,3,128,128), dtype=np.float32) * (10 - i)) + i  # iteration에 따라 분포 변경
        (identity,) = sess.run(None, {'X': data})
        assert np.array_equal(identity, data), "Identity output does not match input!"
        min_val, max_val = data.min(), data.max()
        expected_hist, _ = np.histogram(
            data.flatten(),
            bins=BINS,
            range=(min_val, max_val)
        )
        # 7. fetch stored state from Python binding
        stored_hist = np.array(my_quant_lib.get_histogram(OBS_ID))
        diff = np.abs(stored_hist - expected_hist)
        l1_err        = diff.sum()
        percent_err = l1_err / expected_hist.sum() * 100
        state = my_quant_lib.get_observer_state(OBS_ID)


        abs_max_err = abs(state.max - max_val)
        abs_min_err = abs(state.min - min_val)
        rel_max_err = abs_max_err / (abs(max_val) + 1e-12) * 100  # %
        rel_min_err = abs_min_err / (abs(min_val) + 1e-12) * 100  # %

        print(
            f"Iter {i:3d} | "
            f"hist: {percent_err:.6f}%  | "
            f"Max val: {rel_max_err:.4f}% | "
            f"Min val: {rel_min_err:.4f}%"
        )

