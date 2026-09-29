"""Trace one real frame through the chosen YOLOv8 model (requires inference extra)."""

import argparse
import json
from contextlib import closing
from pathlib import Path

from edgevision.config import load_config
from edgevision.detector import YoloDetector
from edgevision.geometry import preprocess
from edgevision.video import frames

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
args = parser.parse_args()
config = load_config(args.config)
with closing(frames(config.source)) as source:
    frame = next(source)
array, geometry = preprocess(frame.rgb, config.input_size)
detector = YoloDetector(config)
detections = detector.detect(frame.rgb)
assert detector.torch.get_num_threads() == config.threads, "Runtime overrode requested threads"
trace = {}


def shapes(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [shapes(item) for item in value]
    if isinstance(value, dict):
        return {key: shapes(item) for key, item in value.items()}
    return type(value).__name__


def head_inputs(module, inputs):
    trace["head_input_shapes"] = shapes(inputs)


head = detector.model.model.model[-1]
handle = head.register_forward_pre_hook(head_inputs)
try:
    with detector.torch.inference_mode():
        raw = detector.model.model(detector.torch.from_numpy(array))
finally:
    handle.remove()
trace.update(
    frame_shape=list(frame.rgb.shape),
    tensor_shape=list(array.shape),
    tensor_dtype=str(array.dtype),
    geometry=vars(geometry),
    head_type=type(head).__name__,
    raw_output_shapes=shapes(raw),
    detection_count=len(detections),
    actual_torch_threads=detector.torch.get_num_threads(),
    detections=[detection.model_dump() for detection in detections],
)
print(json.dumps(trace, indent=2))
