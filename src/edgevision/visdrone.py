"""Read VisDrone DET ground truth without discarding ignore or class semantics."""

from dataclasses import asdict, dataclass
from pathlib import Path

CLASS_NAMES = (
    "ignored-region",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
)
LABEL_SPACE = "visdrone-det-10-v1"


@dataclass(frozen=True)
class Annotation:
    line_number: int
    xywh: tuple[int, int, int, int]
    score: int
    category_id: int
    truncation: int
    occlusion: int

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        # Source x/y are the top-left corner, unlike YOLO's decoded center/size boxes.
        x, y, width, height = self.xywh
        return x, y, x + width, y + height

    @property
    def role(self) -> str:
        # Preserve the distinction: an ignored region is not an empty background area.
        if self.category_id == 0:
            return "ignored_region"
        if self.category_id == 11:
            return "other"
        return "target" if self.score == 1 else "ignored_object"

    @property
    def class_id(self) -> int | None:
        # Contiguous model IDs apply only to the ten target categories, never COCO IDs.
        return self.category_id - 1 if 1 <= self.category_id <= 10 else None

    def to_dict(self) -> dict:
        return asdict(self) | {
            "xyxy": self.xyxy,
            "category_name": CLASS_NAMES[self.category_id],
            "class_id": self.class_id,
            "role": self.role,
        }


def read_annotations(path: Path) -> tuple[Annotation, ...]:
    """Parse the eight-column DET ground-truth format; errors identify file and line."""
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        # Some files have a trailing comma; additional columns still indicate a wrong format.
        if len(fields) == 9 and fields[-1] == "":
            fields.pop()
        try:
            if len(fields) != 8:
                raise ValueError("expected 8 columns of DET ground truth")
            x, y, width, height, score, category, truncation, occlusion = map(int, fields)
            if width <= 0 or height <= 0:
                raise ValueError("box width and height must be positive")
            if score not in (0, 1):
                raise ValueError("ground-truth score must be 0 or 1, not confidence")
            if not 0 <= category <= 11:
                raise ValueError("category must be in 0..11")
            # Retain -1 as unspecified metadata; never invent an occlusion/truncation value.
            if truncation not in (-1, 0, 1) or occlusion not in (-1, 0, 1, 2):
                raise ValueError("invalid truncation or occlusion code")
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        annotations.append(
            Annotation(line_number, (x, y, width, height), score, category, truncation, occlusion)
        )
    return tuple(annotations)
