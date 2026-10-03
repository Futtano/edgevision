"""Synchronous offline orchestration; bounded memory and no frame dropping."""

import hashlib
import json
import platform
from contextlib import closing
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from PIL import Image, ImageDraw

from edgevision.config import InferenceConfig
from edgevision.detector import Detector, YoloDetector
from edgevision.records import FrameResult
from edgevision.video import frames


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(config: InferenceConfig, detector: Detector | None = None) -> dict:
    # Never overwrite evidence from an earlier run, including a partial failed run.
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=False)
    session_id = str(uuid4())
    summary = {"session_id": session_id, "status": "starting", "frames": 0}
    # Run timing includes setup/hashing, processing, output, and resource cleanup.
    started = perf_counter()
    try:
        # Requested settings with defaults/paths resolved; distinct from artifact identity.
        write_json(output / "config.json", config.model_dump(mode="json"))
        model_hash = sha256(config.weights)
        packages = {}
        for name in ("edgevision", "av", "numpy", "pillow", "ultralytics", "torch"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        # Provenance: content hashes and environment identify what actually ran.
        write_json(
            output / "manifest.json",
            {
                "session_id": session_id,
                "source_sha256": sha256(config.source),
                "model_sha256": model_hash,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": packages,
                "code_sha256": {
                    path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
                },
                "label_space": "coco80",
                "preprocessing": "Pillow bilinear RGB letterbox, padding 114, BCHW /255",
            },
        )
        active_detector = detector if detector is not None else YoloDetector(config)
        if config.save_overlays:
            (output / "overlays").mkdir()
        summary["stop_reason"] = "eof"
        # Equivalent to nested with blocks: close stream first, then flush/close sink.
        # closing() calls generator.close() even after break or a consumer exception,
        # unwinding the video reader's with block while it is suspended at yield.
        with (
            (output / "detections.jsonl").open("w") as sink,
            closing(frames(config.source)) as stream,
        ):
            # Each iteration resumes the generator to decode/convert one frame, then pauses it.
            for frame in stream:
                # Measure only detect(); decode is already timed by the frame generator.
                start = perf_counter()
                detections = active_detector.detect(frame.rgb)
                detector_ms = (perf_counter() - start) * 1000
                height, width, _ = frame.rgb.shape
                result = FrameResult(
                    session_id=session_id,
                    frame_index=frame.index,
                    source_time_s=frame.source_time_s,
                    ingest_monotonic_s=frame.ingest_monotonic_s,
                    width=width,
                    height=height,
                    model_sha256=model_hash,
                    detections=detections,
                    decode_ms=frame.decode_ms,
                    detector_ms=detector_ms,
                )
                # JSONL writes one independent record per frame, including empty detections.
                # Earlier records survive handled failures without buffering the whole run.
                sink.write(result.model_dump_json() + "\n")
                if config.save_overlays:
                    image = Image.fromarray(frame.rgb)
                    draw = ImageDraw.Draw(image)
                    for detection in detections:
                        draw.rectangle(detection.xyxy, outline="lime", width=2)
                        draw.text(
                            detection.xyxy[:2],
                            f"{detection.class_name} {detection.confidence:.2f}",
                            fill="lime",
                        )
                    image.save(output / "overlays" / f"{frame.index:06d}.jpg")
                summary["frames"] += 1
                if config.max_frames is not None and summary["frames"] >= config.max_frames:
                    summary["stop_reason"] = "max_frames"
                    break
        summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        raise
    finally:
        # Run outcome and duration, also on failure; excludes this final summary write.
        summary["elapsed_s"] = perf_counter() - started
        write_json(output / "summary.json", summary)
    return summary
