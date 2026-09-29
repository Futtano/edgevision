"""Portable results: no framework tensors escape the detector boundary."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Detection(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    session_id: str
    frame_index: Annotated[int, Field(ge=0)]
    source_time_s: float | None = Field(default=None, allow_inf_nan=False)
    ingest_monotonic_s: Nonnegative
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    model_sha256: str
    label_space: Literal["coco80"] = "coco80"
    status: Literal["ok"] = "ok"
    detections: list[Detection]
    decode_ms: Nonnegative
    detector_ms: Nonnegative

    @model_validator(mode="after")
    def boxes_within_frame(self):
        for detection in self.detections:
            if detection.xyxy[2] > self.width or detection.xyxy[3] > self.height:
                raise ValueError("Detection extends beyond the original frame")
        return self
