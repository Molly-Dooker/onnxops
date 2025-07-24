# test_histogram_observer.py

import os
import onnx
import onnxruntime as ort
import numpy as np
import my_quant_lib
from numpy.random import default_rng

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
    H = onnx.helper.make_tensor_value_info('H', onnx.TensorProto.INT64, [bins])
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, shape)

    node = onnx.helper.make_node(
        'HistogramObserver',
        inputs=['X'],
        outputs=['H', 'Y'],
        domain='com.my-quant-lib',
        id=observer_id,
        bins=bins
    )

    graph = onnx.helper.make_graph(
        [node],
        'histogram-observer-graph',
        [X],
        [H, Y]
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
    model_path = create_histogram_model(OBS_ID, bins=BINS)

    # 2. register the histogram observer in Python
    my_quant_lib.register_histogram_observer(OBS_ID, BINS)

    # 3. create ONNX Runtime session (CPU EP to ensure state update)
    so = ort.SessionOptions()
    lib_path = os.path.join(os.path.dirname(my_quant_lib.__file__), "libmy_quant_ops.so")
    so.register_custom_ops_library(lib_path)
    sess = ort.InferenceSession(model_path, so, providers=["CUDAExecutionProvider"])

    # 4. generate a random input and run the model
    rng = default_rng(12345)
    data = rng.random((1000,), dtype=np.float32) * 5 + 2  # values in [2,7)
    # reshape to match 2D expectation, e.g. (1000,1)
    data = data.reshape(-1, 1)

    hist_out, identity = sess.run(None, {'X': data})

    # 5. verify identity output equals input
    assert np.array_equal(identity, data), "Identity output does not match input!"

    # 6. compute expected histogram via numpy
    # use same min/max as computed in the node
    min_val, max_val = data.min(), data.max()
    expected_hist, _ = np.histogram(
        data.flatten(),
        bins=BINS,
        range=(min_val, max_val)
    )

    print("ONNX histogram:", hist_out.tolist())
    print("Expected hist :", expected_hist.tolist())
    assert np.array_equal(hist_out, expected_hist), "Histogram mismatch!"

    # 7. fetch stored state from Python binding
    stored_hist = my_quant_lib.get_histogram(OBS_ID)
    print("Stored state hist:", stored_hist)
    assert stored_hist == expected_hist.tolist(), "StateManager histogram mismatch!"

    print("✅ HistogramObserver test passed!")
