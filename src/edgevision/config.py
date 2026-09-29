"""One validated configuration; paths are relative to its YAML file."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Path
    weights: Path
    output_dir: Path
    input_size: Annotated[int, Field(ge=32, le=2048, multiple_of=32)] = 320
    confidence: Probability = 0.25
    nms_iou: Probability = 0.7
    max_detections: Annotated[int, Field(gt=0, le=1000)] = 300
    max_frames: Annotated[int, Field(gt=0)] | None = None
    threads: Annotated[int, Field(ge=1, le=16)] = 4
    device: Literal["cpu"] = "cpu"
    save_overlays: bool = False

    @model_validator(mode="after")
    def local_files(self):
        for path in (self.source, self.weights):
            if not path.is_file():
                raise ValueError(f"Local file does not exist: {path}")
        if self.weights.suffix != ".pt":
            raise ValueError("This adapter expects a local .pt detection checkpoint")
        return self


def load_config(path: Path) -> InferenceConfig:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for key in ("source", "weights", "output_dir"):
        value = raw.get(key)
        if isinstance(value, str):
            raw[key] = (path.parent / value).resolve()
    return InferenceConfig.model_validate(raw)
