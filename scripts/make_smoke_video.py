"""Repeat a still image to test video plumbing. This is NOT a perception benchmark."""

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("data/smoke/bus.mp4"))
args = parser.parse_args()
if args.output.exists():
    parser.error(f"Refusing to overwrite {args.output}")
args.output.parent.mkdir(parents=True, exist_ok=True)
with Image.open(args.image) as source:
    image = source.convert("RGB")
    image.thumbnail((1280, 720))
    # yuv420p requires even dimensions.
    image = image.crop((0, 0, image.width // 2 * 2, image.height // 2 * 2))
    rgb = np.asarray(image)
with av.open(str(args.output), "w") as output:
    stream = output.add_stream("mpeg4", rate=10)
    stream.width, stream.height = image.size
    stream.pix_fmt = "yuv420p"
    for _ in range(8):
        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
print(f"Created {args.output}: 8 repeated-image frames at 10 FPS; smoke test only")
