# 02 — Trust the data before trusting the score

Status: in progress. Step 02A (one image and its annotation contract) is implemented and tested on synthetic data. Dataset acquisition, dataset-wide auditing, conversion, split integrity, and evaluation remain open. No VisDrone accuracy has been measured.

## Question and prerequisites

Module 01 established that pixels can become usable predictions. Module 02 asks: **are those predictions being compared with the right labels, coordinates, and examples?** A working pipeline can still produce a misleading score if its class mapping or evaluation data is wrong.

Read [Module 01](01-inference.md) first. Its geometry, immutable records, provenance, and validation concepts carry over here. The learner confirmed understanding of that walkthrough and its exercises on 2026-10-03.

We will work through three small steps within the [Module 02 roadmap gate](../roadmap.md#02--make-data-and-evaluation-trustworthy):

| Step | Learning question | Status |
| --- | --- | --- |
| 02A: one annotation pair | What does each field mean, and what information must survive parsing? | Implemented; start here |
| 02B: dataset identity and splits | Which exact examples belong to each experiment, and how do we prevent leakage? | Planned |
| 02C: evaluation | How do matching, ignored regions, precision/recall, and AP affect the score? | Planned |

This first step uses tiny synthetic fixtures on CPU, no model downloads or training, and no new dependencies. Allow roughly one learning session to read the code and inspect the outputs; this is a planning estimate.

## What we built

```text
image + VisDrone DET ground-truth text
  → parse and validate rows
  → retain source categories, flags, and attributes
  → derive corner coordinates and target class IDs
  → audit image bounds and duplicate rows
  → JSON records, content hashes, report, and labeled overlay
```

Read these files in order:

1. [visdrone.py](../../src/edgevision/visdrone.py): source annotation contract and derived fields.
2. [data_audit.py](../../src/edgevision/data_audit.py): one-image checks and output artifacts.
3. [cli.py](../../src/edgevision/cli.py): the `audit-image` command.
4. [test_visdrone.py](../../tests/test_visdrone.py): malformed rows, ignore preservation, bounds, duplicate rows, empty labels, and CLI outcomes.

The command is an annotation audit, not a detector. Its boxes come from the annotation file. `status: ok` means the implemented checks found no issues; it does not prove that an annotation describes the correct object or that a dataset is leakage-free.

## Follow one annotation

VisDrone DET uses eight columns:

```text
left,top,width,height,score,category,truncation,occlusion
100,80,60,40,1,4,0,1
```

The example describes a car whose box starts at `(100, 80)` and measures `60×40`. Our derived `xyxy` is `(100, 80, 160, 120)`. These are original-image pixel coordinates; no resizing or letterboxing occurs in this audit.

In ground truth, `score` is a 0/1 inclusion flag, not the detector's confidence. Category 4 means car. Truncation describes cutting off at the image boundary, while occlusion describes covering by another object. The example has truncation 0 and occlusion 1. See the [official DET format](https://github.com/VisDrone/VisDrone2018-DET-toolkit#det-submission-format).

The ten target categories have native IDs 1–10. Our versioned label space, `visdrone-det-10-v1`, maps them to contiguous model IDs 0–9 by subtracting one. Category 0 denotes ignored regions; category 11 denotes others. These receive no target model ID. All source rows remain in the audit, including target-category objects with score 0.

**A class ID only has meaning within its label space.** Our car has native category 4 and model class 3. Module 01 still uses COCO-80; it cannot consume this mapping without a separate model/data integration step. `Annotation` remains distinct from the inference `Detection` record.

> **Optional — four roles, without prematurely implementing evaluation**
>
> The parser derives `target` for category 1–10 with score 1, `ignored_object` for those categories with score 0, `ignored_region` for category 0, and `other` for category 11. It retains score, category, and attributes independently, so no information is lost. An ignored object still has a target class ID, but its role prevents treating it as a normal target. Merely dropping non-target rows would lose information needed by the evaluator. Step 02C must verify the official overlap/matching behavior before reporting metrics; this audit does not implement it.

> **Optional — validation and coordinate policy**
>
> The current ground-truth parser accepts eight integer fields, blank lines, and one optional trailing comma. Width/height must be positive; scores and category/attribute codes are checked. It preserves `-1` for unspecified truncation/occlusion. Predictions with fractional confidence and the longer video-task row format are outside this reader's contract.
>
> Conversion uses continuous edges `(left, top, left + width, top + height)`, consistent with our pipeline. Out-of-image coordinates are retained and reported, rather than silently clipped. The official evaluator may have its own pixel/overlap conventions; parity with it remains a later gate. An empty annotation file is valid; a missing annotation file is an error.

## Run the synthetic experiment

Use the existing locked environment from Module 01. From the repository root, create our own tiny image and labels:

```bash
.venv/bin/python scripts/make_annotation_demo.py
```

This creates `data/synthetic/annotation-demo/scene.png` and `scene.txt`. The gray rectangles are synthetic shapes, not actual VisDrone imagery or model predictions. The helper refuses to reuse an existing directory; skip this command if the fixture already exists.

Audit the pair into a fresh output directory:

```bash
.venv/bin/edgevision audit-image \
  --image data/synthetic/annotation-demo/scene.png \
  --annotations data/synthetic/annotation-demo/scene.txt \
  --split synthetic \
  --output-dir artifacts/runs/module02-annotation-demo
```

All command paths resolve from the shell directory. `--split` records the caller's declaration; it does not verify membership in an official split. Use a new output directory for each run.

Open the outputs:

| File | Inspect |
| --- | --- |
| `annotations.json` | All original fields plus derived coordinates, class ID, and role |
| `manifest.json` | Image/annotation hashes, parser/audit code hashes, class mapping, and declared split |
| `report.json` | Dimensions, role/class counts, duplicate/out-of-bounds findings, and status |
| `overlay.png` | Green targets, yellow ignored objects, orange ignored regions, cyan others |

The expected report has five rows: two targets, one ignored object, one ignored region, and one other. The target counts are one car and one pedestrian. All boxes should remain in the JSON and overlay.

> **Optional — provenance and completion**
>
> Raw inputs remain untouched. Their hashes tie the audit to exact content, while code hashes identify the parser and audit implementation. Keep generated outputs outside Git. `report.json` is written last; an output directory without a report is incomplete. Bounds or duplicate findings produce `status: needs_review` and CLI exit code 1 after saving the report. Structural parsing errors fail before output creation. This single-image manifest is not yet a dataset manifest or a provenance record for an acquired archive.

## Failure and debugging

The tests deliberately introduce a box `(50, 30, 30, 30)` into a `64×48` image, then duplicate its row. Its far corner is `(80, 60)`, outside the frame. The audit records two bounds findings and one duplicate finding while preserving both original rows. Silently clipping would hide the source-data issue.

Run the focused checks:

```bash
.venv/bin/pytest -q tests/test_visdrone.py
```

The tests also distinguish empty labels from missing files and reject malformed rows with the annotation path and source line number. These are injected synthetic failures, not claims about real VisDrone data.

## Design tradeoffs and lessons

We preserve native annotations before exporting a convenient training format. This keeps ignored regions, attributes, and label identities available when evaluation and training policies are added. See [decision 0003](../decisions/0003-preserve-visdrone-annotations.md).

A training-format export, dataset-wide scanner, and evaluator would introduce several contracts at once. Starting with one image makes the semantic boundary inspectable. No change to the Module 01 detector is needed yet.

To check your understanding, follow the score-0 car through `read_annotations()`: why does it retain class ID 3 while receiving the role `ignored_object`? Then inspect why category 0 must not become model class `-1`.

## Evidence and remaining gates

Observed on 2026-10-03: the synthetic command above produced the expected five rows and no audit issues. The overlay was inspected and displayed all four roles. Local artifacts are in `artifacts/runs/module02-annotation-demo/`; their manifest records input and implementation hashes. The fixture uses no randomness. This establishes plumbing and annotation handling, not real-data quality or accuracy.

Validation passed: all 46 CPU tests, Ruff lint/format checks, and ty over application source. Tests and type checking also passed in the locked base environment without PyTorch or Ultralytics. The run used Python 3.12 and the project lockfile, with Module 02 changes on top of source revision `e94776ffbf7255ef5e96fe31dd5b35b43520de65`. Exact parser/audit content hashes are in the run manifest. The synthetic annotation SHA-256 is `fe1960b59bb5739e7a15afb6f9ca8b4d1ac807864132c27e2ff1cd52407ff66d`.

Step 02B will acquire data through the [official dataset listing](https://github.com/VisDrone/VisDrone-Dataset), record archive origin/checksum and applicable terms, and preserve official split directories. It will add deterministic development subsets, image/annotation pairing checks, counts and size statistics, duplicate checks across splits, and a dataset card. Exact duplicates are only one form of leakage; sequence-related and near-duplicate images need additional review. Training uses train; model/threshold selection uses validation; labeled held-out data stays reserved for final reporting. DET/MOT overlap must be checked before combining tasks.

Step 02C will cover IoU, class-aware matching, precision/recall, AP, ignored-region behavior, and sanity cases with known outcomes. We will compare against an identified reference evaluator before making official-score claims. The [evaluation protocol](../evaluation.md) remains the contract for that work.

The module gate remains open until real-data audits, validated conversions/manifests, split checks, and evaluator sanity checks are complete. No downloaded dataset, fine-tuning, aerial accuracy, or completed learner exercise is claimed by this first step.
