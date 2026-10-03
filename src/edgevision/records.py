"""Portable results: no framework tensors escape the detector boundary."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Detection(BaseModel):
    # Pydantic class settings (not JSON fields): reject unknown inputs and reassignment.
    # A validated detection is a fixed observation; its fields contain no mutable lists.
    model_config = ConfigDict(extra="forbid", frozen=True)

    xyxy: tuple[float, float, float, float]
    class_id: Annotated[int, Field(ge=0)]
    class_name: str
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @model_validator(mode="after")
    def valid_box(self):
        x1, y1, x2, y2 = self.xyxy
        if not all(math.isfinite(v) for v in self.xyxy):
            raise ValueError("Box coordinates must be finite")
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("Box must have positive area and nonnegative coordinates")
        return self


class FrameResult(BaseModel):
    # Preserve the validated frame snapshot; later assignments would bypass validation.
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    session_id: str
    frame_index: Annotated[int, Field(ge=0)]
    # Media timeline: presentation timestamp × time base; absent if unavailable.
    source_time_s: float | None = Field(default=None, allow_inf_nan=False)
    # Local perf_counter() after decode/RGB conversion, with an arbitrary clock origin.
    # Compare only with the same local clock, never with source_time_s or wall time.
    ingest_monotonic_s: Nonnegative
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    model_sha256: str
    label_space: Literal["coco80"] = "coco80"
    status: Literal["ok"] = "ok"
    # frozen=True is shallow: a tuple also prevents edits to the collection itself.
    # Pydantic accepts the adapter's list; JSON still serializes this as an array.
    detections: tuple[Detection, ...]
    # Duration of advancing the decoder and converting to RGB; excludes opening video.
    decode_ms: Nonnegative
    # Entire detect() call: preprocessing, prediction/postprocessing, restoration,
    # and validation; first call includes lazy setup. Excludes JSON/overlay output.
    detector_ms: Nonnegative

    @model_validator(mode="after")
    def boxes_within_frame(self):
        for detection in self.detections:
            if detection.xyxy[2] > self.width or detection.xyxy[3] > self.height:
                raise ValueError("Detection extends beyond the original frame")
        return self
