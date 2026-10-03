"""Create synthetic shapes and VisDrone-format labels, not actual VisDrone imagery."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic/annotation-demo"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
image = Image.new("RGB", (640, 360), (32, 32, 32))
draw = ImageDraw.Draw(image)
rows = [
    (100, 80, 60, 40, 1, 4, 0, 1),  # car: target
    (220, 100, 12, 28, 1, 1, 0, 0),  # pedestrian: target
    (300, 40, 180, 100, 0, 0, -1, -1),  # ignored region
    (60, 220, 50, 50, 1, 11, 0, 0),  # others: outside the ten target classes
    (400, 240, 80, 50, 0, 4, 0, 2),  # ignored car, retain its class and attributes
]
for x, y, width, height, *_ in rows:
    draw.rectangle((x, y, x + width - 1, y + height - 1), fill=(100, 100, 100))
image.save(args.output_dir / "scene.png")
(args.output_dir / "scene.txt").write_text(
    "\n".join(",".join(map(str, row)) for row in rows) + "\n", encoding="utf-8"
)
print(f"Created synthetic image and labels in {args.output_dir}; not VisDrone data.")
