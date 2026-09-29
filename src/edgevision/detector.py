"""The only module that knows about Ultralytics; imports stay optional for CPU CI."""

from typing import Protocol

import numpy as np

from edgevision.config import InferenceConfig
from edgevision.geometry import preprocess
from edgevision.records import Detection


class Detector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    def __init__(self, config: InferenceConfig):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(config.threads)
        self.torch = torch
        self.config = config
        self.model = YOLO(str(config.weights), task="detect")
        # select_device() resets CPU threads during lazy predictor setup. This
        # callback runs after setup and before processing each predict invocation.
        self.model.add_callback("on_predict_start", lambda _: torch.set_num_threads(config.threads))
        if self.model.task != "detect":
            raise ValueError("Only bounding-box detection checkpoints are supported")
        if len(self.model.names) != 80 or self.model.names[0] != "person":
            raise ValueError("Module 01 expects the pretrained COCO-80 label space")

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        array, geometry = preprocess(rgb, self.config.input_size)
        with self.torch.inference_mode():
            result = self.model.predict(
                source=self.torch.from_numpy(array),
                imgsz=self.config.input_size,
                device="cpu",
                conf=self.config.confidence,
                iou=self.config.nms_iou,
                max_det=self.config.max_detections,
                verbose=False,
                save=False,
                rect=False,
            )[0]
        # Tensor input is already letterboxed. Vendor boxes refer to that square,
        # not to the original video. NMS has already been applied by predict().
        detections = []
        for row in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, class_id = row
            box = geometry.restore((x1, y1, x2, y2))
            if box is not None:
                detections.append(
                    Detection(
                        xyxy=box,
                        class_id=int(class_id),
                        class_name=self.model.names[int(class_id)],
                        confidence=float(score),
                    )
                )
        return detections
