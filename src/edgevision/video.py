"""Decode sequentially with real presentation timestamps and explicit decoder errors."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import av
import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    source_time_s: float | None
    ingest_monotonic_s: float
    decode_ms: float
    rgb: np.ndarray


def frames(path: Path) -> Generator[Frame, None, None]:
    # Calling frames() creates a generator; opening/decoding starts on first next().
    # Generator exposes close(), letting the consumer release this context on early exit.
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"Source has no video stream: {path}")
        decoder = iter(container.decode(video=0))
        index = 0
        while True:
            # Start a fresh decode interval on each request; excludes opening the video.
            start = perf_counter()
            try:
                frame = next(decoder)
            except StopIteration:
                if index == 0:
                    raise ValueError("Source contains no decodable video frames") from None
                return
            rgb = frame.to_ndarray(format="rgb24")
            # End decode timing and record when the RGB frame became available locally.
            ingest = perf_counter()
            # Pause here while the pipeline detects/writes; that work is not decode time.
            yield Frame(
                index=index,
                source_time_s=float(frame.pts * frame.time_base)
                if frame.pts is not None and frame.time_base is not None
                else None,
                ingest_monotonic_s=ingest,
                decode_ms=(ingest - start) * 1000,
                rgb=rgb,
            )
            index += 1
