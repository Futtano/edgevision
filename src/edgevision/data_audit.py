"""Audit one image/annotation pair before attempting dataset-wide conversion."""

from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from edgevision.pipeline import sha256, write_json
from edgevision.visdrone import CLASS_NAMES, LABEL_SPACE, read_annotations

SPLITS = ("train", "val", "test-dev", "synthetic")
COLORS = {"target": "lime", "ignored_object": "yellow", "ignored_region": "orange", "other": "cyan"}


def audit_image(image_path: Path, annotation_path: Path, split: str, output_dir: Path) -> dict:
    """Preserve source rows; flag geometry for review rather than silently clipping it."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    annotations = read_annotations(annotation_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")  # Fully decode so corrupted image data can fail here.
    width, height = image.size
    issues = []
    seen = set()
    for annotation in annotations:
        x1, y1, x2, y2 = annotation.xyxy
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            issues.append({"line": annotation.line_number, "kind": "out_of_bounds"})
        row = (
            annotation.xywh,
            annotation.score,
            annotation.category_id,
            annotation.truncation,
            annotation.occlusion,
        )
        if row in seen:
            issues.append({"line": annotation.line_number, "kind": "duplicate_row"})
        seen.add(row)
    counts = Counter(annotation.role for annotation in annotations)
    report = {
        "schema_version": "0.1",
        "status": "needs_review" if issues else "ok",
        "scope": "single_image_annotation_audit",
        "declared_split": split,  # Caller metadata; one pair cannot verify split integrity.
        "label_space": LABEL_SPACE,
        "width": width,
        "height": height,
        "annotation_count": len(annotations),
        "counts_by_role": {role: counts[role] for role in COLORS},
        "target_counts_by_class": {
            name: sum(a.role == "target" and a.category_id == category for a in annotations)
            for category, name in enumerate(CLASS_NAMES)
            if 1 <= category <= 10
        },
        "issues": issues,
    }
    # Content identities bind the report to exact inputs; raw sources remain untouched.
    manifest = {
        "image_path": str(image_path.resolve()),
        "image_sha256": sha256(image_path),
        "annotation_path": str(annotation_path.resolve()),
        "annotation_sha256": sha256(annotation_path),
        "declared_split": split,
        "label_space": LABEL_SPACE,
        "target_class_map": {str(i - 1): CLASS_NAMES[i] for i in range(1, 11)},
        "parser_sha256": sha256(Path(__file__).with_name("visdrone.py")),
        "audit_sha256": sha256(Path(__file__)),
    }
    # Match inference's no-overwrite rule, even for empty or incomplete prior runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "annotations.json", {"annotations": [a.to_dict() for a in annotations]})
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        color = COLORS[annotation.role]
        draw.rectangle(annotation.xyxy, outline=color, width=2)
        draw.text(
            annotation.xyxy[:2],
            f"{annotation.line_number}: {CLASS_NAMES[annotation.category_id]} ({annotation.role})",
            fill=color,
        )
    image.save(output_dir / "overlay.png")
    # Write the report last: an output directory alone is not proof of completion.
    write_json(output_dir / "report.json", report)
    return report
