from pathlib import Path

import av
import numpy as np
import pytest

from edgevision.config import InferenceConfig


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=10)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "bgr0"
        for index in range(4):
            rgb = np.zeros((48, 64, 3), dtype=np.uint8)
            rgb[:, :, 0] = index * 50
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


@pytest.fixture
def config(tmp_path: Path, clip: Path) -> InferenceConfig:
    weights = tmp_path / "test.pt"
    weights.write_bytes(b"Not a model; tests inject a detector")
    return InferenceConfig(source=clip, weights=weights, output_dir=tmp_path / "run")
