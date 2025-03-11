from onnxruntime_extensions import get_library_path
import onnxruntime as ort
import importlib.resources

_options = ort.SessionOptions()
_options.inter_op_num_threads, _options.intra_op_num_threads = 1, 1
_options.register_custom_ops_library(get_library_path())
_providers = ["CPUExecutionProvider"]

use_model = ort.InferenceSession(
    path_or_bytes = str(importlib.resources.files('memory').joinpath('resources/use_quantized.onnx')),
    sess_options=_options,
    providers=_providers
)

def get_onnx_embeddings(texts):
    return use_model.run(output_names=["outputs"], input_feed={"inputs": texts})[0].tolist()
