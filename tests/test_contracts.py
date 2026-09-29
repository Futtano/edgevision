from pathlib import Path

import pytest
from pydantic import ValidationError

from edgevision.config import InferenceConfig, load_config
from edgevision.records import Detection, FrameResult


@pytest.mark.parametrize(
    "change",
    [
        {"input_size": 321},
        {"confidence": float("nan")},
        {"device": "cuda"},
        {"max_frames": 0},
        {"threadz": 4},
        {"threads": 0},
    ],
)
def test_invalid_configuration_fails_before_execution(config, change):
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(config.model_dump() | change)


def test_paths_are_relative_to_yaml_not_cwd(config, tmp_path, monkeypatch):
    config_file = tmp_path / "inference.yaml"
    config_file.write_text(f"source: {config.source.name}\nweights: test.pt\noutput_dir: results\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = load_config(config_file)
    assert result.source == config.source
    assert result.output_dir == tmp_path / "results"


def test_missing_source_rejected(config):
    with pytest.raises(ValidationError, match="does not exist"):
        InferenceConfig.model_validate(config.model_dump() | {"source": Path("missing.mp4")})


@pytest.mark.parametrize("box", [(0, 0, 0, 1), (-1, 0, 1, 1), (0, 0, float("inf"), 1)])
def test_invalid_detection_box(box):
    with pytest.raises(ValidationError):
        Detection(xyxy=box, class_id=0, class_name="person", confidence=0.5)


def test_results_reject_out_of_frame_coordinates():
    detection = Detection(xyxy=(0, 0, 100, 50), class_id=0, class_name="person", confidence=0.5)
    with pytest.raises(ValidationError, match="beyond"):
        FrameResult(
            session_id="test",
            frame_index=0,
            ingest_monotonic_s=1,
            width=64,
            height=48,
            model_sha256="test",
            detections=[detection],
            decode_ms=0,
            detector_ms=0,
        )
