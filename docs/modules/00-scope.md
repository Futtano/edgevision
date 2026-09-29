# 00 — Scope and budgets

Status: in progress. Hardware and availability are user-provided; no performance measurements have been run.

## Question

What useful aerial-perception system can we operate within the available machine, time, and training budget? See the [module 00 gate](../roadmap.md).

## Known constraints

| Item | Current information |
| --- | --- |
| Machine | Minisforum V3 3-in-1 notebook |
| CPU | AMD Ryzen 7 8840U |
| Graphics | Integrated Radeon 780M |
| RAM | 32 GB |
| Learning time | Up to 10 hours/week |
| External compute | Free or low-cost GPU/spot sessions are possible |
| Cloud budget | No provider or numerical spending limit established |
| OS/runtime and free storage | To inventory before environment setup |

## Initial design consequences

CPU PyTorch and ONNX Runtime are the reference local paths. The integrated GPU is not a CUDA device; other acceleration backends require separate compatibility and performance checks. TensorRT is outside the core local deployment scope.

Use local CPU runs for data validation, inference, tests, service behavior, and tiny training sanity checks. Plan meaningful fine-tuning around external GPU availability. Allow separate CPU and training dependency profiles if accelerator dependencies require them.

Keep model input size separate from source-video size. Resizing reduces compute but can lose small-object detail; measure that tradeoff. Set throughput and freshness targets after the first local benchmark.

For interruptible training, validate save/resume in a short run before a full run. Preserve model, optimizer/scheduler state, progress, RNG state where supported, resolved config, and data identity in persistent storage. Document what the selected framework restores. Evaluate exported artifacts locally: cloud GPU speed is not local deployment speed.

## Remaining completion work

- Inventory OS/runtime, CPU threads, and free disk space.
- Choose the reference clip, source rate/resolution, and initial classes.
- Draft stage budgets, throughput/freshness targets, and memory limits; validate them in module 01.
- Establish maximum cloud spend and per-run duration before any paid session; free-only remains a valid initial constraint.
- Specify checkpoint persistence and compute/storage cleanup.

## What we have established

Learning and implementation can start locally without waiting for a cloud GPU. Training compute and deployment compute are separate design choices. No claims about achievable FPS or training duration have been measured.
