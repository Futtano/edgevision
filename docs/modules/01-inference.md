# 01 — Follow one frame through inference

Status: implementation and smoke checks complete. The walkthrough and learner exercise below are ready; this does not imply that the learner has completed them. No training, aerial evaluation, or real-time benchmark has been performed.

## The question

How does a video frame become detections that a different component can safely use? The first boundary is between **pixels and predictions**. Getting it right requires explicit image layout, geometry, labels, and timing—not just a successful model call.

Read [the prerequisite concepts](../prerequisites.md) and [scope and budgets](00-scope.md) first. The prerequisite page gives you the array, box, pipeline, timing, and project vocabulary used here; CNN training details come later.

## What we built

```text
local video
  → PyAV decode to RGB uint8 HWC
  → letterbox to a square
  → float32 RGB BCHW tensor in [0, 1]
  → pretrained YOLOv8n + confidence filtering + NMS
  → inverse resize/padding and clip to original frame
  → validated detections → JSONL and optional JPEG overlays
```

The path is synchronous: each frame finishes before the next is decoded. Offline processing slows the source instead of dropping frames. Only one frame and its results are retained by our loop. There is no service, tracker, or live queue yet.

Read the code in this order:

1. [video.py](../../src/edgevision/video.py): frames and source presentation timestamps.
2. [geometry.py](../../src/edgevision/geometry.py): the actual preprocessing and inverse coordinate transform.
3. [detector.py](../../src/edgevision/detector.py): the framework adapter and canonical detection output.
4. [records.py](../../src/edgevision/records.py): invalid boxes/scores fail visibly.
5. [pipeline.py](../../src/edgevision/pipeline.py): orchestration, results, and run provenance.

[config.py](../../src/edgevision/config.py) validates configuration before execution. Unknown settings, missing local files, invalid probabilities, and unsupported devices fail early. YAML paths resolve relative to the YAML file, not the shell's current directory. Each run needs a new output directory.

## Setup and first run

Run from the repository root. These commands target our Linux/WSL CPU environment. The lockfile selects CPU PyTorch wheels; model inference is an optional extra so CI can remain small.

```bash
uv sync --locked --extra inference --python 3.12
mkdir -p artifacts/models data/smoke .cache/ultralytics
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

curl --fail --location \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt \
  --output artifacts/models/yolov8n.pt

.venv/bin/python scripts/make_smoke_video.py \
  --image .venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg

.venv/bin/edgevision infer --config configs/inference.yaml \
  --output-dir artifacts/runs/my-first-run
```

The smoke-video helper refuses to replace an existing video. If the prepared video and weights already exist, skip acquisition and generation. Choose a fresh output directory for each inference run. `--output-dir` resolves relative to the current shell directory, unlike paths inside YAML.

The checkpoint comes from the [official Ultralytics asset release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). The example image ships with the locked Ultralytics package. The generated video repeats that still image eight times at 10 FPS; it tests decoding, geometry, inference, and EOF, not motion or aerial accuracy. Keep these assets out of Git. Dataset and distribution licensing decisions for the full project remain separate; see the [Ultralytics license options](https://www.ultralytics.com/license).

For the prepared workspace, inspected results already exist under `artifacts/runs/module01-verified/`:

- `detections.jsonl`: one record per decoded frame, including empty successful results.
- `overlays/000000.jpg`: boxes drawn on the original-resolution image.
- `config.json`: resolved configuration.
- `manifest.json`: input/model hashes, source module hashes, package and platform versions.
- `summary.json`: completion/failure status, processed count, stop reason, and elapsed time.
- `frame-trace.json`: the separately generated model-shape walkthrough.

A failed run retains its partial results and a failed summary; downstream consumers must check that summary. The `max_frames` limit is reported separately from EOF. PyAV propagates decoder errors, although a container that decodes successfully is not proof that the source file is complete or uncorrupted.

## Follow the geometry by hand

Our smoke frame has **height 720, width 540**, with three RGB channels. At target size 320:

```text
scale = min(320 / 540, 320 / 720) = 4/9
resized width  = 240
resized height = 320
left padding  = (320 - 240) / 2 = 40
 top padding  = 0
```

The frame changes from `(720, 540, 3)` uint8 to `(1, 3, 320, 320)` float32. The dimensions mean batch, channels, height, width. Dividing by 255 changes the range to `[0, 1]`; it does not change RGB channel order. PyAV explicitly emits RGB here, avoiding an implicit OpenCV BGR assumption.

For a continuous-edge box `(x1, y1, x2, y2)`, preprocessing transforms each x coordinate as `x * resized_width / width + left`, and each y coordinate similarly with height and top. To restore a predicted box:

```text
x_original = (x_model - left) * width / resized_width
y_original = (y_model - top)  * height / resized_height
```

Use actual rounded resize dimensions for each axis. On odd-sized images, integer rounding means the two effective scales can differ slightly. Clip to the original frame after inversion. A detection entirely in padding has zero visible area and is discarded; NaN or inverted model boxes raise errors.

**Try this before running anything:** for a 1280×720 landscape frame resized to 320, where does the original box `(400, 200, 800, 600)` land? The answer is `(100, 120, 200, 220)`: the scale is 1/4 and top padding is 70. Explain why multiplying all four model coordinates by four would produce the wrong y coordinates. The [geometry tests](../../tests/test_geometry.py) exercise this case and rounded portrait/landscape cases.

## Follow the model internals

Generate the trace with:

```bash
.venv/bin/python scripts/inspect_frame.py --config configs/inference.yaml
```

The inspection script checks the effective thread count and attaches a temporary PyTorch hook to the detection head. On the locked YOLOv8n checkpoint, we observed:

| Representation | Shape | Meaning |
| --- | --- | --- |
| Input | `1 × 3 × 320 × 320` | One RGB image |
| Head feature map 1 | `1 × 64 × 40 × 40` | Stride 8 spatial features |
| Head feature map 2 | `1 × 128 × 20 × 20` | Stride 16 spatial features |
| Head feature map 3 | `1 × 256 × 10 × 10` | Stride 32 spatial features |
| Decoded prediction tensor | `1 × 84 × 2100` | Four box values and 80 class scores per candidate |

There are `40² + 20² + 10² = 2100` candidate positions. Those are not 2,100 final objects. The prediction path filters by confidence and applies non-maximum suppression (NMS) to reduce duplicate boxes. This checkpoint's raw decoded box values are center/size form; the adapter receives postprocessed `xyxy` boxes from `predict()`. We do not apply NMS a second time.

The trace also contains `boxes: [1, 64, 2100]` in the head's auxiliary output: the 64 channels encode four sets of 16 localization bins before distribution decoding, not 64 classes. We will study the loss and assignment machinery in module 03. The present purpose is to distinguish features, candidate predictions, and final detections.

Because we pass an already-preprocessed tensor into Ultralytics, its returned coordinates refer to our square input. Our adapter owns the inverse transform. If we instead passed the original image through Ultralytics' image path, that path would own preprocessing and restoration; applying our inverse again would be wrong. The input-layout distinction is documented in [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/).

The canonical label space is **COCO-80**, including `person` and `bus`. It is not VisDrone, and class IDs must not be carried into that dataset without explicit mapping. The smoke model is a temporary learning reference rather than the final aerial detector choice.

The detector protocol lets the pipeline use any object with a compatible `detect()` method. The YOLO adapter imports its optional libraries only when needed and returns our own detection records. Its Ultralytics `predict()` return annotation includes both lists and iterators, plus embedding tensors: we request `stream=False`, consume the first result with `next(iter(predictions))`, and check that it is a `Results` object containing boxes. This resolves the iterator-indexing type error without suppressing it. Tests simulate those Ultralytics return variants without installing the inference dependencies; empty boxes remain a valid successful prediction, while a missing result or wrong result type raises a clear error.

## Experiment, observation, and debugging

Hypothesis: an explicitly letterboxed RGB tensor plus inverse geometry should yield valid boxes aligned with the original image. Eight repeated-image frames should be read in order, and the run should terminate at EOF even though the config permits 30 frames.

Observed: eight records, frame indices 0–7, source times 0.0–0.7 s, and a successful EOF summary. The first frame contains three person detections and one bus detection. The first overlay was visually inspected; the boxes align with the original scene. Synthetic tests cover empty detections, invalid coordinates, frame limits, and partial failure. See the [smoke evidence](../reports/module01-smoke.md) for identifiers and timings.

A real integration issue appeared during inspection: Ultralytics' `select_device()` resets PyTorch CPU threads during lazy predictor setup. Setting four threads before constructing YOLO did not preserve the setting. We now reapply it through `on_predict_start`, after setup and before prediction. The inspection script verifies the effective value is four. In the installed Ultralytics 8.4.166 predictor, this callback runs before the first warmup and inference; warmup still contributes to first-call overhead. The thread setting applies to PyTorch CPU operations across the process, not separately to each detector instance.

The initial environment also selected Python 3.13 despite the project's 3.12 requirement. Explicitly selecting `/usr/bin/python3` (verified as 3.12.3) fixed setup. The repository includes `.python-version`; use the explicit interpreter if your shell/environment overrides it. This is why a lockfile and a Python version both matter. The CPU-wheel source follows the [uv PyTorch integration pattern](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Timing semantics and limits

The video reader returns a `Generator[Frame, None, None]`: calling it does not open the video until iteration begins, and the type exposes its `close()` method. The pipeline uses `closing()` to unwind the reader's video context on early frame limits or consumer failures; a plain `Iterator` annotation hid that method from the type checker. Decode timing surrounds only decoder advancement/RGB conversion; the generator pauses during detection and output. Run timing additionally includes setup, hashing, output, and resource cleanup, but excludes the final summary write.

`Detection` and `FrameResult` are frozen validated snapshots: field reassignment is rejected, and unknown input fields are forbidden. Since Pydantic freezing is shallow, `FrameResult.detections` is also a tuple, preventing collection edits that could bypass the frame-boundary validator. The adapter can still supply a list, which Pydantic converts; JSON output remains an array. Source-list mutation, rejected reassignment, and JSON round-trip compatibility are covered by a CPU contract test.

`decode_ms` includes decoding and conversion to an RGB array. `detector_ms` includes our preprocessing, Ultralytics prediction/postprocessing, restoration, and detection validation. It excludes JSON serialization and overlays; the first value also includes lazy runtime setup. It is not pure neural-network forward time.

`source_time_s` derives from presentation timestamp × time base, not `frame_index / FPS`; it can be absent. `ingest_monotonic_s` is host time after decoding. Subtracting those two clock domains would not measure latency. Full pipeline throughput, sustained frame age, memory ceilings, and p95/p99 need the later benchmark protocol.

## Completion evidence and next learning step

Type checking is now a development dependency (`ty==0.0.84`) recorded in `uv.lock` and run in CI with `uv run --locked ty check`. The initial configured scope is application code under `src`; tests and inspection scripts remain covered by Ruff and runtime tests rather than this type-checking gate. Base CI omits optional inference packages, so only the three PyTorch/Ultralytics imports in the detector adapter may be unresolved. When the inference extra is installed, ty uses those packages' types normally. See [ty import configuration](https://docs.astral.sh/ty/reference/configuration/#allowed-unresolved-imports). Use the locked environment for regular checks rather than fetching ty through `uvx`.

The CPU suite verifies geometry, validation, real video decoding with synthetic frames, orderly EOF, explicit frame limits, existing-output protection, and failed-run evidence. Run:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/ty check
.venv/bin/pytest -q
```

The adapter typing update on 2026-10-03 passed all 32 tests and `ty check src/edgevision/detector.py --python .venv/bin/python` (ty was run with `uvx`). The new tests cover list/iterator predictions, tensor/NumPy box storage, empty boxes, and invalid Ultralytics results without inference dependencies. An eight-frame real-model smoke run also completed at EOF; its config, manifest, predictions, and summary are saved locally under `artifacts/runs/module01-adapter-types/`, run ID `5083fcdf-7272-42e4-92a7-f75ad25a2260`. These are integration checks, not new performance benchmarks.

A fresh environment was also used to repeat real-model inference from the lockfile; see the report. No GPU or dataset/model downloads are required by CI tests. The learner's immediate next step is to inspect one JSON record and its overlay, work through the coordinate exercise, and explain where the 2,100 candidates come from. Module 02 then replaces convenient demo assumptions with audited aerial data and trustworthy evaluation.
