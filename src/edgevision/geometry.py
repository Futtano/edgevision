"""Explicit letterboxing and inverse geometry, shared by tests and inference."""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int

    def restore(self, xyxy) -> tuple[float, float, float, float] | None:
        """Undo rounded resize/padding, clip continuous edges, drop padding-only boxes."""
        coords = np.asarray(xyxy, dtype=np.float64)
        if coords.shape != (4,) or not np.isfinite(coords).all():
            raise ValueError("Expected four finite box coordinates")
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            raise ValueError("Detector produced a nonpositive-area box")
        # Remove the padding offset, then undo each axis's resize to recover source coordinates.
        coords[[0, 2]] = (coords[[0, 2]] - self.left) * self.width / self.resized_width
        coords[[1, 3]] = (coords[[1, 3]] - self.top) * self.height / self.resized_height
        # Keep only the visible part of boxes that extend into padding or beyond the frame.
        coords[[0, 2]] = coords[[0, 2]].clip(0, self.width)
        coords[[1, 3]] = coords[[1, 3]].clip(0, self.height)
        x1, y1, x2, y2 = map(float, coords)
        # A box entirely in padding clips to zero area; None means no visible detection remains.
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def preprocess(rgb: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """RGB uint8 HWC → letterboxed RGB float32 BCHW in [0, 1]."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected a uint8 RGB image with shape H×W×3")
    height, width, _ = rgb.shape
    # YOLOv8 downsamples five times by 2 (maximum stride 32), then merges
    # feature maps after 2x upsampling. For size=320, 10x10 upsamples to
    # 20x20 and matches the finer map. For size=321, rounded sizes are
    # 11x11 and 21x21: upsampling gives 22x22, so concatenation cannot match.
    # This requirement is for the padded model input, not the source frame.
    if min(width, height, size) <= 0 or size % 32:
        raise ValueError("Positive dimensions and an input size divisible by 32 are required")
    gain = min(size / width, size / height)
    # Extreme aspect ratios can round the shorter dimension to zero; keep at least one pixel.
    resized_width = max(1, round(width * gain))
    resized_height = max(1, round(height * gain))
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    # Bilinear resampling blends nearby pixel values along both axes, separately per channel.
    # Pillow widens the contributing neighborhood when shrinking an image.
    resized = Image.fromarray(rgb).resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[top : top + resized_height, left : left + resized_width] = np.asarray(resized)
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, Letterbox(width, height, resized_width, resized_height, left, top)
