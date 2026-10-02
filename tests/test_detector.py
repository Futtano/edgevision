"""Exercise the Ultralytics adapter boundary without importing PyTorch or downloading weights."""

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from edgevision.detector import YoloDetector


@pytest.fixture
def adapter(monkeypatch, config):
    # Supply only the Ultralytics result type needed by detect(); no inference extra in CI.
    class Results:
        def __init__(self, rows=None):
            self.boxes = None if rows is None else SimpleNamespace(data=Rows(rows))

    class Rows:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32).reshape(-1, 6)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    for name in ("ultralytics", "ultralytics.engine", "ultralytics.engine.results"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(
        sys.modules["ultralytics.engine.results"], "Results", Results, raising=False
    )
    detector = YoloDetector.__new__(YoloDetector)
    detector.config = config
    detector.torch = SimpleNamespace(
        inference_mode=nullcontext, from_numpy=lambda array: array, Tensor=Rows
    )
    detector.model = SimpleNamespace(names={0: "person"})
    return detector, Results


@pytest.mark.parametrize("container", [list, iter])
@pytest.mark.parametrize("backing", ["tensor", "numpy"])
def test_prediction_list_and_iterator_restore_same_box(adapter, container, backing):
    detector, Results = adapter

    def predict(**kwargs):
        assert kwargs["stream"] is False
        assert kwargs["source"].shape == (1, 3, 320, 320)
        result = Results([[100, 20, 160, 80, 0.9, 0]])
        if backing == "numpy":
            result.boxes.data = result.boxes.data.numpy()
        return container([result])

    detector.model.predict = predict
    detections = detector.detect(np.zeros((720, 540, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].xyxy == pytest.approx((135, 45, 270, 180))
    assert detections[0].class_name == "person"


@pytest.mark.parametrize("kind", ["missing_result", "embedding", "missing_boxes"])
def test_invalid_ultralytics_return_raises_explicit_error(adapter, kind):
    detector, Results = adapter
    responses = {
        "missing_result": [],
        "embedding": [np.zeros(8)],
        "missing_boxes": [Results()],
    }
    detector.model.predict = lambda **kwargs: iter(responses[kind])
    with pytest.raises(ValueError, match="Detector returned"):
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))


def test_empty_detection_boxes_are_a_valid_result(adapter):
    detector, Results = adapter
    detector.model.predict = lambda **kwargs: [Results([])]
    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []
