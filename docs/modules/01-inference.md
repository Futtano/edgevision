# 01 — Follow one frame through inference

Status: implementation and smoke checks complete. On 2026-10-03, the learner confirmed understanding of the walkthrough, exercises, and pipeline structure. No training, aerial evaluation, or real-time benchmark has been performed.

## The question

How does a video frame become detections that a different component can safely use? The first boundary is between **pixels and predictions**. Getting it right requires explicit image layout, geometry, labels, and timing—not just a successful model call.

Read [the prerequisite concepts](../prerequisites.md) and [scope and budgets](00-scope.md) first. The prerequisite page gives you the array, box, pipeline, timing, and project vocabulary used here; CNN training details come later.

Follow the main text and commands for the working path. Alerts marked **Optional** preserve additional explanations for later review; you can skip them on a first pass. Backbone, neck, and multi-scale prediction-branch architecture will be studied in Module 03.

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

[config.py](../../src/edgevision/config.py) validates configuration before execution. Unknown settings, missing local files, invalid probabilities, and unsupported devices fail early.

YAML paths resolve relative to the YAML file; the CLI output override resolves relative to the shell directory. Choose an output directory that does not already exist.

> **Optional — path resolution and preserving evidence**
>
> Relative paths inside YAML resolve from the directory containing that YAML file. For example, `source: ../data/smoke/bus.mp4` in `configs/inference.yaml` points to the repository's `data/smoke/bus.mp4`: `..` goes up from `configs/`. The same configuration selects the same files regardless of the shell's current directory, making runs predictable when launched from another directory or a script. Absolute paths remain absolute. The CLI's `--output-dir` override instead resolves from the current shell directory, following normal command-line conventions.
>
> Each run requires an output directory that does not already exist. Its configuration, manifest, predictions, summary, and overlays form one experiment record. Reusing a directory could overwrite earlier evidence or leave stale overlays from a longer previous run. The pipeline therefore rejects existing directories, including empty directories and partial failed runs; failures remain available for debugging. Choose a fresh name for subsequent runs, such as `--output-dir artifacts/runs/experiment-02` when running from the repository root.

## Setup and first run

Run from the repository root: the shell paths below assume that location. These commands target our Linux/WSL CPU environment. The lockfile selects CPU PyTorch wheels; model inference is an optional extra so CI can remain small. A trailing `\` continues a command onto the next line.

**1. Synchronize the project environment.**

```bash
uv sync --locked --extra inference --python 3.12
```

> **Optional — environment command options**
>
> `sync` makes `.venv` match the selected project dependencies. `--locked` requires the lockfile to match the project specification and fails if it needs updating. `--extra inference` includes optional PyTorch/Ultralytics dependencies; the development group (pytest, Ruff, and ty) is installed by default. `--python 3.12` selects a Python 3.12 interpreter.

**2. Prepare local directories.**

```bash
mkdir -p artifacts/models data/smoke .cache/ultralytics
```

> **Optional — directory command options**
>
> These hold model weights, the smoke video, and Ultralytics settings. `-p` creates missing parents and accepts directories that already exist. These assets and settings stay outside Git.

**3. Set the Ultralytics settings location.**

```bash
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
```

Repeat this export in a new shell before inference.

> **Optional — shell variable and settings directory**
>
> `$PWD` expands to the current directory. Quotes preserve paths containing spaces, and `export` makes the variable available to programs launched from this shell. This directs Ultralytics settings to the project's ignored cache directory for this shell session; repeat it in a new shell before running inference.

**4. Download the pretrained checkpoint, if it is not already present.**

```bash
curl --fail --location \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt \
  --output artifacts/models/yolov8n.pt
```

Skip this download if the weights already exist; curl can overwrite its destination.

> **Optional — download command options**
>
> `--fail` returns an error for HTTP failure responses such as 404. `--location` follows redirects used by GitHub release downloads. `--output` saves the weights at the named path; this command can overwrite that file, so skip it when the prepared weights already exist. Downloading a checkpoint does not train the model.

**5. Generate the smoke video, if it is not already present.**

```bash
.venv/bin/python scripts/make_smoke_video.py \
  --image .venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg
```

This creates `data/smoke/bus.mp4`: eight repeated-image frames at 10 FPS, with no motion or aerial evaluation. Skip generation if the video already exists; the helper refuses to overwrite it.

> **Optional — smoke helper options**
>
> `.venv/bin/python` runs the helper with the synchronized project's interpreter and packages, without activating the environment. `--image` selects the example image bundled with Ultralytics. The helper's default output is `data/smoke/bus.mp4`; `--output` can change it. It writes eight repeated-image frames at 10 FPS and refuses to overwrite an existing video. This checks decoding, geometry, inference, and EOF; it does not evaluate motion or aerial accuracy.

**6. Run inference into a fresh output directory.**

```bash
.venv/bin/edgevision infer --config configs/inference.yaml \
  --output-dir artifacts/runs/my-first-run
```

This loads the YAML and overrides its output directory. Use a new directory name for each run.

> **Optional — inference command options**
>
> `edgevision` is the CLI entry point installed from our package into `.venv/bin`. `infer` selects inference, `--config` loads the YAML settings, and `--output-dir` overrides the configured output path. That override resolves relative to the current shell directory, unlike paths inside YAML. Choose a different directory name for another run; an existing directory is rejected.

> **Optional — using uv run instead**
>
> **Why use `.venv/bin` rather than `uv run`?** The commands above directly use the environment synchronized in step 1, without another dependency synchronization. You can instead use the following equivalents, which check/synchronize the environment before execution. Keep `--extra inference` explicit: a later `uv run` does not automatically remember the extra selected by an earlier `uv sync`. `--locked` prevents silent lockfile updates.
>
> ```bash
> uv run --locked --extra inference python scripts/make_smoke_video.py \
>   --image .venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg
>
> uv run --locked --extra inference edgevision infer \
>   --config configs/inference.yaml \
>   --output-dir artifacts/runs/my-first-run
> ```
>
> These are alternatives to steps 5 and 6, not additional runs; the same overwrite protections apply.

The checkpoint comes from the [official Ultralytics asset release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). The example image ships with the locked Ultralytics package. Dataset and distribution licensing decisions for the full project remain separate; see the [Ultralytics license options](https://www.ultralytics.com/license).

For the prepared workspace, inspected results already exist under `artifacts/runs/module01-verified/`:

- `detections.jsonl`: one record per decoded frame, including empty successful results.
- `overlays/000000.jpg`: boxes drawn on the original-resolution image.
- `config.json`: resolved configuration.
- `manifest.json`: input/model hashes, source module hashes, package and platform versions.
- `summary.json`: completion/failure status, processed count, stop reason, and elapsed time.

> **Optional — why the run has several output files**
>
> Configuration answers “what settings did we request?”; the manifest answers “which exact inputs, model, code, and environment did we use?” A path alone cannot identify an artifact whose contents may change, so the manifest records hashes. Predictions answer “what did the system detect?”; the summary answers “how did the run end?” Overlays support visual inspection. JSONL stores one independent JSON object per line, allowing incremental writes without holding every prediction in memory. The manifest identifies artifacts but does not contain the video or weights themselves.

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

> **Optional — rounding, clipping, and invalid boxes**
>
> Use actual rounded resize dimensions for each axis. On odd-sized images, integer rounding means the two effective scales can differ slightly. Clip to the original frame after inversion. A detection entirely in padding has zero visible area and is discarded; NaN or inverted model boxes raise errors.

> **Optional — preprocessing implementation details**
>
> `min(size / width, size / height)` fits the longer dimension to the square while preserving aspect ratio. `max(1, round(...))` prevents an extreme aspect ratio from rounding the shorter dimension to zero. Bilinear resizing blends nearby pixel values along both axes, separately per channel; Pillow widens the contributing neighborhood when shrinking. The resized image replaces the center of a gray RGB array filled with 114. Transposing HWC to CHW changes axis order; adding `[None]` supplies the batch axis. The result is initially a contiguous float32 NumPy array, converted to a PyTorch tensor inside the adapter.
>
> The square model input must be divisible by 32 for this YOLOv8 adapter, because downsampling and upsampling must produce compatible feature-map sizes for merging. At input 320, stride-16/32 grids are 20/10; doubling 10 gives 20. At input 321, those grids can become 21/11; doubling 11 gives 22, which does not match 21. Original video dimensions need not be divisible by 32.

**Try this before running anything:** for a 1280×720 landscape frame resized to 320, where does the original box `(400, 200, 800, 600)` land? The answer is `(100, 120, 200, 220)`: the scale is 1/4 and top padding is 70. Explain why multiplying all four model coordinates by four would produce the wrong y coordinates. The [geometry tests](../../tests/test_geometry.py) exercise this case and rounded portrait/landscape cases.

## Follow the model internals

The pipeline does not produce `frame-trace.json`. The separate [inspection script](../../scripts/inspect_frame.py) prints a trace to stdout. Inspect it with:

```bash
.venv/bin/python scripts/inspect_frame.py --config configs/inference.yaml
```

> **Optional — saving the inspection trace**
>
> Redirect stdout to save the trace, using an existing parent directory and a fresh filename:
>
> ```bash
> .venv/bin/python scripts/inspect_frame.py --config configs/inference.yaml \
>   > artifacts/runs/my-first-run/frame-trace.json
> ```
>
> This is a separate inspection artifact, not a normal pipeline output. Shell `>` overwrites an existing file. The previously inspected smoke directory also contains a trace generated separately this way.

The inspection script checks the effective thread count and attaches a temporary PyTorch hook to the detection head. On the locked YOLOv8n checkpoint, we observed:

| Representation | Shape | Meaning |
| --- | --- | --- |
| Input | `1 × 3 × 320 × 320` | One RGB image |
| Head feature map 1 | `1 × 64 × 40 × 40` | Stride 8 spatial features |
| Head feature map 2 | `1 × 128 × 20 × 20` | Stride 16 spatial features |
| Head feature map 3 | `1 × 256 × 10 × 10` | Stride 32 spatial features |
| Decoded prediction tensor | `1 × 84 × 2100` | Four box values and 80 class scores per candidate |

The three grids supply `40² + 20² + 10² = 2100` candidates, each with four box values and 80 class scores. Confidence filtering and NMS reduce these to final detections. Our adapter receives `xyxy` boxes and does not apply NMS again.

> **Optional — prediction tensor and box decoding**
>
> The decoded tensor's axes are `[batch, values per candidate, candidate positions]`:
>
> - `1`: one frame in the batch.
> - `84 = 4 + 80`: four box values plus a score for each COCO class. This YOLOv8 output has no separate objectness channel. Dimension index 1 describes each candidate; it is not a spatial image axis.
> - `2100`: one candidate per spatial position across the three scales. At input size 320, strides 8, 16, and 32 give grids `40×40`, `20×20`, and `10×10`, contributing `1600 + 400 + 100` candidates. Their spatial axes are flattened and the candidate axes concatenated.
>
> For candidate `k`, `prediction[0, :, k]` selects its 84 values:
>
> ```text
> [cx, cy, width, height, person_score, bicycle_score, …]
> ```
>
> `cx, cy` are the box's **center**, not its top-left corner. These decoded coordinates describe the square model input. Conversion to corner coordinates uses:
>
> ```text
> x1 = cx - width / 2      y1 = cy - height / 2
> x2 = cx + width / 2      y2 = cy + height / 2
> ```
>
> For example, center/size values `(150, 170, 100, 100)` become model-space `xyxy = (100, 120, 200, 220)`. For the landscape example above, removing top padding 70 and undoing the 1/4 scale then restores the original box `(400, 200, 800, 600)`. Converting box format and restoring image coordinates are separate operations.
>
> Those 2,100 candidates are not 2,100 final objects. Ultralytics converts center/size boxes to `xyxy`, filters by confidence, and applies non-maximum suppression (NMS) to reduce duplicate boxes. The adapter receives those postprocessed corner coordinates from `predict()`, then restores original-frame coordinates. We do not apply NMS a second time. Here, “decoded” refers to interpreting the model's box representation, independently of video decoding.
>
> The trace also contains `boxes: [1, 64, 2100]` in the head's auxiliary output: the 64 channels encode four sets of 16 localization bins before distribution decoding, not 64 classes. We will study the loss and assignment machinery in module 03. The present purpose is to distinguish features, candidate predictions, and final detections.

> **Optional — who owns coordinate restoration**
>
> Because we pass an already-preprocessed tensor into Ultralytics, its returned coordinates refer to our square input. Our adapter owns the inverse transform. If we instead passed the original image through Ultralytics' image path, that path would own preprocessing and restoration; applying our inverse again would be wrong. The input-layout distinction is documented in [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/).

The canonical label space is **COCO-80**, including `person` and `bus`. It is not VisDrone, and class IDs must not be carried into that dataset without explicit mapping. The smoke model is a temporary learning reference rather than the final aerial detector choice.

> **Optional — adapter contract and prediction return types**
>
> The detector protocol lets the pipeline use any object with a compatible `detect()` method. The YOLO adapter imports its optional libraries only when needed and returns our own detection records. Its Ultralytics `predict()` return annotation includes both lists and iterators, plus embedding tensors: we request `stream=False`, consume the first result with `next(iter(predictions))`, and check that it is a `Results` object containing boxes. This resolves the iterator-indexing type error without suppressing it. Tests simulate those Ultralytics return variants without installing the inference dependencies; empty boxes remain a valid successful prediction, while a missing result or wrong result type raises a clear error.

## Experiment, observation, and debugging

Hypothesis: an explicitly letterboxed RGB tensor plus inverse geometry should yield valid boxes aligned with the original image. Eight repeated-image frames should be read in order, and the run should terminate at EOF even though the config permits 30 frames.

Observed: eight records, frame indices 0–7, source times 0.0–0.7 s, and a successful EOF summary. The first frame contains three person detections and one bus detection. The first overlay was visually inspected; the boxes align with the original scene. Synthetic tests cover empty detections, invalid coordinates, frame limits, and partial failure. See the [smoke evidence](../reports/module01-smoke.md) for identifiers and timings.

> **Optional — runtime thread setting and interpreter debugging**
>
> A real integration issue appeared during inspection: Ultralytics' `select_device()` resets PyTorch CPU threads during lazy predictor setup. Setting four threads before constructing YOLO did not preserve the setting. We now reapply it through `on_predict_start`, after setup and before prediction. The inspection script verifies the effective value is four. In the installed Ultralytics 8.4.166 predictor, this callback runs before the first warmup and inference; warmup still contributes to first-call overhead. The thread setting applies to PyTorch CPU operations across the process, not separately to each detector instance.
>
> The initial environment also selected Python 3.13 despite the project's 3.12 requirement. Explicitly selecting `/usr/bin/python3` (verified as 3.12.3) fixed setup. The repository includes `.python-version`; use the explicit interpreter if your shell/environment overrides it. This is why a lockfile and a Python version both matter. The CPU-wheel source follows the [uv PyTorch integration pattern](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Timing semantics and limits

`decode_ms` and `detector_ms` measure separate stages; their sum excludes serialization and overlays. Source time and local ingest time belong to different clock domains.

> **Optional — generator cleanup and immutable records**
>
> The video reader returns a `Generator[Frame, None, None]`: calling it does not open the video until iteration begins, and the type exposes its `close()` method. The pipeline uses `closing()` to unwind the reader's video context on early frame limits or consumer failures; a plain `Iterator` annotation hid that method from the type checker. Decode timing surrounds only decoder advancement/RGB conversion; the generator pauses during detection and output. Run timing additionally includes setup, hashing, output, and resource cleanup, but excludes the final summary write.
>
> `Detection` and `FrameResult` are frozen validated snapshots: field reassignment is rejected, and unknown input fields are forbidden. Since Pydantic freezing is shallow, `FrameResult.detections` is also a tuple, preventing collection edits that could bypass the frame-boundary validator. The adapter can still supply a list, which Pydantic converts; JSON output remains an array. Source-list mutation, rejected reassignment, and JSON round-trip compatibility are covered by a CPU contract test.

> **Optional — the timing sequence for one run**
>
> Run-start reading → configuration/hashing/model setup → first generator request opens the video → decode-start reading → decode/RGB conversion → ingest reading → yield → detector-start reading → `detect()` → detector-end reading → record validation/JSON/overlay output. The next iteration resumes the generator and starts a new decode interval. At EOF or early exit, resources close before the final run-time reading; writing `summary.json` follows that reading. The final unsuccessful decode attempt at EOF produces no frame timing record. Decode and detector variables named `start` live in separate function scopes.
>
> `perf_counter()` measures elapsed time, including waiting and scheduling delays, rather than CPU execution time alone. Its arbitrary absolute value is useful only through differences on the same clock. See [Python timing](https://docs.python.org/3/library/time.html#time.perf_counter) and [generator cleanup with closing](https://docs.python.org/3/library/contextlib.html#contextlib.closing).

`decode_ms` includes decoding and conversion to an RGB array. `detector_ms` includes our preprocessing, Ultralytics prediction/postprocessing, restoration, and detection validation. It excludes JSON serialization and overlays; the first value also includes lazy runtime setup. It is not pure neural-network forward time.

`source_time_s` derives from presentation timestamp × time base, not `frame_index / FPS`; it can be absent. `ingest_monotonic_s` is host time after decoding. Subtracting those two clock domains would not measure latency. Full pipeline throughput, sustained frame age, memory ceilings, and p95/p99 need the later benchmark protocol.

## Completion evidence and next learning step

Development tools are locked alongside application dependencies; CI runs the checks below without model/data downloads.

> **Optional — type-checking scope and optional imports**
>
> Type checking is now a development dependency (`ty==0.0.84`) recorded in `uv.lock` and run in CI with `uv run --locked ty check`. The initial configured scope is application code under `src`; tests and inspection scripts remain covered by Ruff and runtime tests rather than this type-checking gate. Base CI omits optional inference packages, so only the three PyTorch/Ultralytics imports in the detector adapter may be unresolved. When the inference extra is installed, ty uses those packages' types normally. See [ty import configuration](https://docs.astral.sh/ty/reference/configuration/#allowed-unresolved-imports). Use the locked environment for regular checks rather than fetching ty through `uvx`.

The CPU suite verifies geometry, validation, real video decoding with synthetic frames, orderly EOF, explicit frame limits, existing-output protection, and failed-run evidence. Run:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/ty check
.venv/bin/pytest -q
```

> **Optional — historical adapter validation evidence**
>
> The adapter typing update on 2026-10-03 passed all 32 tests and `ty check src/edgevision/detector.py --python .venv/bin/python` (ty was run with `uvx`). The new tests cover list/iterator predictions, tensor/NumPy box storage, empty boxes, and invalid Ultralytics results without inference dependencies. An eight-frame real-model smoke run also completed at EOF; its config, manifest, predictions, and summary are saved locally under `artifacts/runs/module01-adapter-types/`, run ID `5083fcdf-7272-42e4-92a7-f75ad25a2260`. These are integration checks, not new performance benchmarks.

A fresh environment was also used to repeat real-model inference from the lockfile; see the report. No GPU or dataset/model downloads are required by CI tests. The learner's immediate next step is to inspect one JSON record and its overlay, work through the coordinate exercise, and explain where the 2,100 candidates come from. Module 02 then replaces convenient demo assumptions with audited aerial data and trustworthy evaluation.
