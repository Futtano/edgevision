import json

import numpy as np
import pytest

from edgevision.pipeline import run
from edgevision.records import Detection
from edgevision.video import frames


class ExampleDetector:
    def detect(self, rgb):
        # Emit an empty result for the first (black) frame.
        if rgb.max() == 0:
            return []
        return [Detection(xyxy=(1, 2, 20, 30), class_id=0, class_name="person", confidence=0.9)]


def test_decode_preserves_order_rgb_and_source_timestamps(clip):
    decoded = list(frames(clip))
    assert [frame.index for frame in decoded] == [0, 1, 2, 3]
    assert [frame.source_time_s for frame in decoded] == pytest.approx([0, 0.1, 0.2, 0.3])
    np.testing.assert_array_equal(decoded[1].rgb[0, 0], [50, 0, 0])


def test_pipeline_eof_empty_frames_and_artifacts(config):
    summary = run(config.model_copy(update={"save_overlays": True}), ExampleDetector())
    assert summary["status"] == "completed" and summary["stop_reason"] == "eof"
    assert summary["frames"] == 4
    rows = [
        json.loads(line)
        for line in (config.output_dir / "detections.jsonl").read_text().splitlines()
    ]
    assert [row["frame_index"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["detections"] == [] and rows[0]["status"] == "ok"
    assert rows[1]["detections"][0]["xyxy"] == [1, 2, 20, 30]
    assert len(list((config.output_dir / "overlays").glob("*.jpg"))) == 4
    manifest = json.loads((config.output_dir / "manifest.json").read_text())
    assert len(manifest["source_sha256"]) == 64
    assert manifest["model_sha256"] == rows[0]["model_sha256"]


def test_frame_limit_is_reported_separately_from_eof(config):
    summary = run(config.model_copy(update={"max_frames": 2}), ExampleDetector())
    assert summary["frames"] == 2 and summary["stop_reason"] == "max_frames"


def test_existing_output_is_never_overwritten(config):
    config.output_dir.mkdir()
    sentinel = config.output_dir / "keep.txt"
    sentinel.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        run(config, ExampleDetector())
    assert sentinel.read_text() == "previous evidence"


def test_detector_error_records_failed_run_and_preserves_partial_results(config):
    class BrokenDetector:
        def detect(self, rgb):
            if rgb.max() > 0:
                raise RuntimeError("Injected failure")
            return []

    with pytest.raises(RuntimeError, match="Injected failure"):
        run(config, BrokenDetector())
    summary = json.loads((config.output_dir / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["frames"] == 1
    assert summary["error_type"] == "RuntimeError"
    assert len((config.output_dir / "detections.jsonl").read_text().splitlines()) == 1


def test_invalid_video_cannot_be_reported_as_success(config, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not video")
    with pytest.raises(Exception, match="Invalid data"):
        run(config.model_copy(update={"source": bad}), ExampleDetector())
    summary = json.loads((config.output_dir / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["frames"] == 0
