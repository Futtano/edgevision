"""The only module that knows about Ultralytics; imports stay optional for CPU CI."""

from typing import Protocol

import numpy as np

from edgevision.config import InferenceConfig
from edgevision.geometry import preprocess
from edgevision.records import Detection


class Detector(Protocol):
    # Structural contract: any compatible detect() method works, without inheritance.
    def detect(self, rgb: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    # Adapter: keep library outputs and APIs behind the pipeline's Detection contract.
    def __init__(self, config: InferenceConfig):
        # Load optional inference dependencies only when constructing a real detector;
        # pipeline tests can import this module and inject a fake without these packages.
        import torch
        from ultralytics import YOLO

        # Parallel CPU work within tensor operations, not concurrent video frames.
        # This setting affects the whole process; more threads may add contention.
        torch.set_num_threads(config.threads)
        # Retain the locally imported module for detect(); this does not copy PyTorch.
        self.torch = torch
        self.config = config
        self.model = YOLO(str(config.weights), task="detect")
        # Ultralytics select_device() overwrites the thread count during lazy setup.
        # Reapply ours at each prediction's start, after setup and before warmup/inference.
        # The callback receives the predictor as its argument; '_' deliberately ignores it.
        self.model.add_callback("on_predict_start", lambda _: torch.set_num_threads(config.threads))
        if self.model.task != "detect":
            raise ValueError("Only bounding-box detection checkpoints are supported")
        if len(self.model.names) != 80 or self.model.names[0] != "person":
            raise ValueError("Module 01 expects the pretrained COCO-80 label space")

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        from ultralytics.engine.results import Results

        array, geometry = preprocess(rgb, self.config.input_size)
        # Skip gradient bookkeeping for prediction; Ultralytics sets evaluation mode.
        with self.torch.inference_mode():
            predictions = self.model.predict(
                source=self.torch.from_numpy(array),
                stream=False,
                imgsz=self.config.input_size,
                device="cpu",
                conf=self.config.confidence,
                iou=self.config.nms_iou,
                max_det=self.config.max_detections,
                verbose=False,
                save=False,
                rect=False,
            )
            # predict() advertises either a list or iterator, so consume one result
            # through iteration. Our input contains exactly one image.
            try:
                result = next(iter(predictions))
            except StopIteration:
                raise ValueError("Detector returned no result for the input frame") from None
            # The Ultralytics return type also includes embedding tensors; narrow to boxes.
            if not isinstance(result, Results) or result.boxes is None:
                raise ValueError("Detector returned a result without bounding-box predictions")
        # Tensor input is already letterboxed. Ultralytics boxes refer to that square,
        # not to the original video. NMS has already been applied by predict().
        # Results can store boxes as either a PyTorch tensor or a NumPy array.
        box_data = result.boxes.data
        if isinstance(box_data, self.torch.Tensor):
            box_data = box_data.cpu().numpy()
        detections = []
        for row in box_data:
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
