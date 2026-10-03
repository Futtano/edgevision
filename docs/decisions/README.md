# Architecture decisions

Use a short architecture decision record (ADR) when a choice affects component boundaries, data/evaluation semantics, deployment, or future maintenance. Each record contains status, context, decision, alternatives, consequences, and a condition for revisiting it.

## 0001 — Keep the core on one machine

Date: 2026-09-29

Status: accepted as the initial planning baseline; revise with evidence.

**Context:** the project must teach production CV and system design while remaining manageable for one learner. Hardware and weekly availability are not yet known.

**Decision:** one active stream, two detector families, one tracker, one selected deployment runtime, and one API on one machine. Start with recorded video and add paced replay for overload experiments. Separate training/export commands from the inference service. Introduce infrastructure only in the module that needs it.

**Alternatives:** a notebook-only comparison would leave operational behavior unexplored. A distributed multi-camera platform would add infrastructure work before we have measured the inference pipeline.

**Consequences:** we can study contracts, state, queuing, artifacts, telemetry, recovery, and rollback locally. We do not demonstrate distributed scaling, live-network reliability, fleet operations, or flight-system integration. Reference PyTorch and deployment ONNX paths remain available for parity checks.

**Revisit when:** the core release is complete and a measured requirement justifies another stream, machine, or live source.

## Decisions to make during implementation

Module 02's initial data boundary is recorded in [0003 — Preserve native VisDrone annotations](0003-preserve-visdrone-annotations.md). Training export and evaluator parity remain open decisions.

- Exact detector checkpoints, framework versions, licensing, and compute budget — modules 00–03.
- Dataset conversion/ignore policy and held-out evaluation protocol — module 02.
- Deployment model choice based on accuracy, latency, and memory — module 04.
- Runtime precision and export compatibility — module 06.
- Queue policy, frame-gap behavior, and service lifecycle — module 07.
- Promotion/rollback procedure and operating targets — module 08.

## 0002 — Explicit preprocessing and a temporary CPU detector

Date: 2026-09-29

Status: accepted for module 01; final training model choice remains open.

**Context:** we need an inspectable path from video pixels to boxes before adding aerial data and training. Local hardware is CPU-first.

**Decision:** use pretrained YOLOv8n as a small, established study reference. Decode with PyAV to preserve presentation timestamps and surface decoder errors. Implement RGB letterboxing explicitly with Pillow, pass an RGB BCHW tensor to the detector, and own inverse coordinate mapping. Delegate model execution and NMS to Ultralytics. Keep the heavy inference stack optional; lock CPU wheels with uv.

**Alternatives:** passing original images directly to the framework is shorter but hides the transformation we want to study. Reimplementing the entire detector/postprocessor would make the first module too large.

**Consequences:** we can test geometry without weights and replace the model behind a detection contract. Pillow interpolation is part of our preprocessing identity and may differ slightly from a framework image path; training/export comparisons must preserve or explicitly compare it. This model uses COCO labels, not VisDrone. Ultralytics code and checkpoints carry upstream licensing terms; final repository/distribution licensing is not settled by this dependency choice. No license has been assigned to the repository on the user's behalf.

**Revisit when:** module 02 fixes aerial class/evaluation semantics, module 03 selects the trainable baseline, or export parity reveals a preprocessing discrepancy.
