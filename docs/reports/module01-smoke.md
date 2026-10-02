# Module 01 smoke evidence

Date: 2026-09-29. This is a bounded integration check, not an accuracy or throughput benchmark.

## Inputs and provenance

- Source: eight repeated frames of the packaged Ultralytics bus image, encoded with the committed smoke-video helper at 10 FPS, 540×720 pixels.
- Detector: YOLOv8n pretrained COCO checkpoint from the official v8.3.0 assets release.
- Runtime: Python 3.12.3, Ultralytics 8.4.166, PyTorch 2.14.0+cpu, PyAV 16.1.0; complete versions and hashes in the [machine-readable evidence](module01-smoke.json).
- Host: Ryzen 7 8840U, WSL2/Ubuntu 24.04, four PyTorch inference threads, batch one, float32, 320×320 model input.
- Code: module 01 working tree based on `0f57218`; per-module SHA-256 hashes identify the actual implementation at measurement time. The report also records the lockfile hash.

Raw local outputs are under `artifacts/runs/module01-verified/`. They are ignored by Git; the compact JSON evidence preserves timing samples, tensor trace, manifests, and completion summaries for readers of the repository.

## Results

Eight frames were processed in order, with presentation times from 0.0 to 0.7 seconds. EOF was reached before the configured 30-frame limit. Each frame produced three person detections and one bus detection; the first overlay was visually inspected for alignment. This is a ground-level example, not aerial evaluation evidence.

| Observation | Value |
| --- | --- |
| Verified run ID | `545e5b93-098f-4192-84d9-345cbc30e253` |
| First detector call, including lazy setup | 3023.43 ms |
| Median of subsequent seven detector calls | 61.39 ms |
| Subsequent-call range | 49.85–83.24 ms |
| Complete run elapsed time | 6.58 s |
| Actual threads verified by inspection script | 4 |

`detector_ms` includes preprocessing, prediction/postprocessing, inverse geometry, and record validation. It excludes decode, serialization, and overlay output. Run elapsed time also includes imports, file hashing, model loading, and output. There was no prescribed warmup, no controlled competing workload, and no meaningful sample size for tail latency. Do not invert the median and present it as end-to-end FPS. The provisional 5 FPS/500 ms freshness/4 GiB memory goals remain unvalidated.

The initial exploratory run used the thread setting reset by Ultralytics; it remains at `artifacts/runs/first-inference/` but is excluded from this report. Thread count was fixed and verified before collecting the reported run.

## Reproduction and checks

See the [walkthrough](../modules/01-inference.md) for acquisition and environment setup. Commands executed for the reported run:

```bash
YOLO_CONFIG_DIR="$PWD/.cache/ultralytics" .venv/bin/edgevision infer \
  --config configs/inference.yaml --output-dir artifacts/runs/module01-verified
YOLO_CONFIG_DIR="$PWD/.cache/ultralytics" .venv/bin/python scripts/inspect_frame.py \
  --config configs/inference.yaml
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -q
```

Choose new output paths when rerunning; existing evidence is intentionally protected. All 24 tests passed and Ruff lint/format checks passed. The CPU-only test suite also passed before the optional inference dependencies were installed. GitHub Actions is configured but has not been run remotely.

A separate fresh environment was created with the same lockfile:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/edgevision-module01-clean UV_CACHE_DIR="$PWD/.uv-cache" \
  uv sync --locked --extra inference --python /usr/bin/python3
YOLO_CONFIG_DIR="$PWD/.cache/ultralytics" \
  /tmp/edgevision-module01-clean/bin/edgevision infer \
  --config configs/inference.yaml --output-dir artifacts/runs/module01-clean-env
```

That run completed all eight frames at EOF. All frame timestamps, detection counts, and classes matched the primary run; paired boxes differed by less than 0.01 pixels and confidence scores by less than 1e-5. This checks environment reproduction on the same machine and assets, not cross-machine determinism or clean-checkout acquisition on another OS.

## Remaining limits

No annotated dataset, fine-tuning, tracking, ONNX export, service, sustained load, or memory benchmark is involved. Aerial data selection and class/ignore semantics are the next module. The image/model remain local external assets; their upstream terms and provenance apply.
