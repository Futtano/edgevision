# Project roadmap

## Goal and scope

Build a portfolio-quality perception pipeline while learning how its data, models, runtime, and operational boundaries work. Prefer a small system with credible measurements and explained failure modes over a large collection of integrations.

**Core scope:** one aerial-data family, two detector families, one tracker, one machine, one active video stream, one API, and one repeatable release procedure. Keep research/training offline and inference online. Reuse established training and tracking implementations; implement the surrounding contracts, validation, evaluation, and operational behavior ourselves.

**Deferred:** flight hardware, ROS 2, SLAM, planning/control, multimodal fusion, re-identification across cameras, Kubernetes, Kafka, distributed training, automated hyperparameter search, custom CUDA, frontend application, and automated retraining. Revisit an item only when a measured problem or a new learning objective justifies it.

## Working defaults

These are planning choices, not installed dependencies. Pin actual versions and checkpoint identifiers when each component is introduced.

| Concern | Default | Why / boundary |
| --- | --- | --- |
| Environment | Python, PyTorch, uv lockfile, pytest, Ruff, GitHub Actions | One reproducible development path; CPU checks from the first code module |
| Configuration | YAML with validated typed configuration | Start with one resolved config; add no configuration framework unless needed |
| Data | VisDrone DET for detection; MOT sequences for tracking | Preserve task-specific annotations and evaluation rules |
| Detectors | One small YOLO-style convolutional model; one small RT-DETR-family model | Compare architectural families; do not chase every newest release |
| Experiments | Local MLflow plus saved resolved configs | One tracking system; no hosted service required |
| Data versions | Immutable manifests, checksums, acquisition/conversion scripts | Add DVC only if managing multiple revisions becomes burdensome |
| Tracking | ByteTrack through one adapter | Learn association and temporal state without maintaining a new tracker |
| Runtime | PyTorch reference, ONNX Runtime deployment | TensorRT only after a compatible NVIDIA target is available |
| Service | FastAPI, one worker owning the pipeline, server-sent events | Small control API and one-way result stream |
| Operations | Docker, structured logs, Prometheus-format metrics | A dashboard is optional; useful measurements are mandatory |

Choose the precise detector checkpoint after a hardware smoke test and checking the selected code, weights, and dataset terms. Ultralytics is a convenient candidate, but its licensing is part of that choice. The RT-DETR family provides an accessible transformer study; the project does not claim that it is the newest or best detector in every setting. See [references](#references-and-tooling-notes).

## Milestones and effort

Estimates are focused learner hours, including notes and experiments, excluding unattended training. Treat them as planning ranges and revise after the first two modules.

| Module | Focus | Hours | Visible result |
| --- | --- | --- | --- |
| 00 | Requirements and budgets | 3–5 | Scenario, hardware inventory, provisional success criteria |
| 01 | First complete inference path | 6–10 | Recorded clip → pretrained detections → JSON and overlay |
| 02 | Data and evaluation contracts | 8–12 | Audited dataset, manifests, trustworthy evaluator |
| 03 | Reproducible CNN baseline | 10–16 | Fine-tuned detector with traceable run and error report |
| 04 | Transformer internals and comparison | 12–18 | RT-DETR run and controlled comparison |
| 05 | Temporal state and tracking | 8–12 | Tracks, association analysis, sequence metrics |
| 06 | Runtime export and profiling | 8–12 | ONNX artifact, parity check, benchmark report |
| 07 | Streaming and service design | 10–14 | Bounded pipeline, API, overload and recovery behavior |
| 08 | Deployment and observability | 8–12 | Container, telemetry, operational drills |
| 09 | Failure analysis and release | 8–12 | Model/system cards, release report, runnable demo |

Core total: **81–123 focused hours**. At your availability of up to 10 hours/week, plan roughly **9–13 active weeks**, plus contingency for debugging and training access. This is a scope estimate, not a deadline or a GPU-time promise.

Your reference machine is a Minisforum V3 with AMD Ryzen 7 8840U, Radeon 780M integrated graphics, and 32 GB RAM. Start with CPU PyTorch and ONNX Runtime. Integrated-GPU acceleration is an optional compatibility experiment; the AMD GPU is not a CUDA target. Keep CPU training to tiny correctness exercises and plan meaningful fine-tuning on bounded external GPU sessions if available.

No cloud provider or spending limit is selected. Before a cloud run, estimate duration from a smoke run, establish a total budget, verify checkpoint save/resume, and define shutdown/cleanup steps. Use portable scripts and resolved configs for free notebooks or interruptible instances. Preserve checkpoints and run metadata outside ephemeral instance storage. Include transfer and storage costs; free allocations are opportunistic rather than a schedule guarantee.

Suggested cadence: weeks 1–2 for scope/inference/data, weeks 3–6 for training and architecture study, weeks 6–8 for tracking/ONNX, weeks 9–11 for streaming/operations, and weeks 12–13 for release analysis and overruns. Completion gates take precedence over approximate dates.

Milestone A (00–03) is a reproducible detection project. Milestone B (04–06) adds model comparison, tracking, and a deployment runtime. Milestone C (07–09) finishes the operational portfolio project. Finish one module's gate before expanding its scope. Documentation and meaningful tests are part of each module, not a final cleanup phase.

The default learning order is sequential. If training compute is temporarily unavailable, use a clearly labeled pretrained model to work on 05–08; keep the training/comparison gates open and do not present pretrained runs as fine-tuned results.

## 00 — Define the system we can actually build

**Understand:** a model metric versus a product requirement; throughput versus response time; hardware limits; offline evaluation versus live freshness.

**Build:** record OS, CPU/RAM, GPU/VRAM, available storage, weekly learning time, and a maximum training-time or spend budget. Pick one recorded-video scenario, target classes, source resolution/rate, and deployment machine. State whether the initial deployment is CPU or GPU.

**Explore:** sketch a latency budget for decode, resize, inference, postprocessing, tracking, and output. Explain why a 25 ms forward pass does not imply 40 FPS for the complete application. Define what happens if processing cannot keep up.

**Gate:** commit a scenario and provisional budgets for throughput, p95 frame age, memory, and training effort. Targets are hypotheses until measured. Name a CPU fallback if accelerator access is uncertain.

**Record:** `docs/modules/00-scope.md`, initial architecture assumptions, and unresolved hardware choices.

## 01 — Build a thin inference slice and follow one frame

**Understand:** decoded image layout, RGB/BGR, tensor shape and dtype, normalization, resizing/letterboxing, coordinate transforms, confidence thresholds, and postprocessing.

**Build:** package skeleton, locked environment, validated config, local-video reader, a pretrained detector adapter, and JSONL output with an optional overlay. Add lightweight lint and CPU test CI immediately. Use small synthetic fixtures in CI rather than downloading models or datasets.

**Explore:** annotate the shapes and coordinate spaces from a frame to model input and back. Draw a synthetic box, transform it through letterboxing, and recover it. Inspect the selected detector's feature maps, class/box outputs, and any NMS. Pretrained COCO labels are not yet VisDrone labels.

**Design question:** what should callers know about a detector so that replacing the implementation does not change the pipeline?

**Gate:** a short local clip produces frame-indexed results, including empty frames; coordinates align with the original image; end-of-file is clean. Tests catch incorrect inverse transforms and malformed output. A fresh environment can reproduce the command.

**Record:** `docs/modules/01-inference.md`, frame walkthrough, initial timings labeled exploratory, and adapter contract.

## 02 — Make data and evaluation trustworthy

**Understand:** box conventions, class mapping, ignored regions, leakage, imbalance, small-object statistics, IoU, matching, precision/recall, and AP.

**Build:** acquisition instructions; immutable raw-data references; conversion and validation scripts; checksum manifests; official split preservation; a deterministic tiny development subset. Keep source labels and metadata so ignored annotations are not silently lost. Audit invalid boxes, dimensions, duplicates, and class counts. Record provenance and redistribution restrictions.

**Explore:** compute IoU and a tiny precision/recall example by hand; compare it with a reference evaluator. Visualize random annotations and tiny/crowded examples. Deliberately introduce a wrong class mapping and confirm validation catches it.

**Design question:** how can a training run identify exactly which examples and transformation rules it consumed?

**Gate:** validated conversions and split manifests; correct behavior on empty images, ignored regions, and invalid annotations; an evaluator sanity check with known predictions. Use sequence-level grouping for video splits and inspect cross-task overlap before training on DET and evaluating MOT. Reserve labeled held-out data for final reporting.

**Record:** `docs/modules/02-data.md`, a dataset card, and the implemented version of the [evaluation protocol](evaluation.md). Any simplified metric must be labeled as such rather than called an official VisDrone score.

## 03 — Train a reproducible convolutional baseline

**Understand:** backbone/neck/head, multiscale features, localization and classification losses, assignment of training targets, augmentation, transfer learning, and train/eval modes. Inspect the chosen implementation rather than assuming every YOLO is anchor-based or uses identical losses.

**Build:** one config-driven fine-tuning entry point with local MLflow. Save source revision, environment, manifest hash, class map, initialization weights, seeds, resolved settings, logs, and checkpoint. Add checkpoint resume and a bounded smoke-training configuration.

**Explore:** first overfit a tiny subset to validate the learning path. Then compare pretrained and fine-tuned performance. Run one hypothesis-driven ablation, such as input resolution versus small-object recall and latency, with the other settings fixed.

**Design question:** which artifacts are required to reproduce an experiment, and what reproducibility can nondeterministic kernels prevent?

**Gate:** successful small-subset learning; one budgeted baseline run; validation AP/recall by class and object size; an error gallery; verified checkpoint reload/resume; enough artifacts for another person to repeat it. Report seed sensitivity if affordable; otherwise state that results are single-run estimates.

**Record:** `docs/modules/03-baseline.md`, run links/IDs, learning curves, and the first model card. No arbitrary AP threshold is required; unexplained failure to learn blocks completion.

## 04 — Understand a transformer detector and compare fairly

**Understand:** attention, queries/keys/values, spatial/positional information, multi-scale features, object queries, decoder predictions, set prediction, bipartite matching, and losses. Study DETR's formulation before the specific RT-DETR encoder and query-selection optimizations. Matching during training and NMS during inference are distinct operations.

**Build:** a second adapter and one budgeted RT-DETR-family fine-tune. Reuse the data semantics, evaluation pipeline, output contract, and experiment record. Do not recreate a full transformer detector from scratch.

**Explore:** run a tiny attention example, trace actual tensor shapes through one forward pass, and solve a small matching-cost matrix. Explain how a hybrid CNN/transformer detector differs from a classification ViT. Relate feature resolution and query count to small-object behavior and compute.

**Design question:** would you ship the higher-AP model if its p95 latency, memory, or export path violated the deployment requirements?

**Gate:** comparison on the same held-out validation protocol and same benchmark hardware. Record input size, pretraining data, training budget, parameter count, artifact size, memory, AP, recall, and latency. Separate controlled comparisons from each model's recommended recipe; differences in pretraining and tuning are confounders. Select the deployment candidate using an accuracy/performance tradeoff, without assuming the transformer must win.

**Record:** `docs/modules/04-transformers.md`, the comparison report, and an ADR for the selected deployment model.

## 05 — Turn detections into temporal state

**Understand:** tracking-by-detection, motion prediction, Kalman filtering, IoU association, high/low confidence association, track birth/loss/deletion, ID switches, and camera-motion limitations. ByteTrack's use of lower-confidence detections is the main study target. ([ByteTrack implementation](https://github.com/FoundationVision/ByteTrack))

**Build:** one tracker adapter, trajectory output, per-stream state, and MOT evaluation on appropriately labeled sequences. Preserve detection scores needed by association. Make resets, timestamp gaps, and class compatibility explicit.

**Explore:** inspect an occlusion frame by frame; vary one association threshold; insert a frame gap; compare against a tiny greedy IoU tracker used only as a learning reference. Explain a case where detection AP improves but track identity consistency worsens.

**Design question:** who owns state, what resets it, and how does dropping a frame change the motion model's assumptions?

**Gate:** HOTA/IDF1 and ID-switch reporting with documented preprocessing, plus tests for reset, empty detections, class mismatches, and sequence boundaries. Use complete sequences for comparable MOT evaluation. Live dropping is a separately reported operating mode. Verify evaluator support or implement and validate a dataset adapter rather than assuming a format is supported. ([TrackEval](https://github.com/JonathonLuiten/TrackEval))

**Record:** `docs/modules/05-tracking.md`, association walkthrough, sequence-level results, and failure clips.

## 06 — Export, measure, and explain runtime costs

**Understand:** computation graphs, static/dynamic shapes, execution providers, CPU/GPU transfers, synchronization, warmup, precision, and numerical versus task-level parity.

**Build:** export the selected model to ONNX; validate its metadata and preprocessing; run the same pipeline through ONNX Runtime. Add a repeatable benchmark command with raw timing output and a runtime manifest. Profile the whole pipeline before optimizing it.

**Explore:** compare PyTorch and ONNX at the same resolution, precision, and batch size. Separate decode, preprocessing, transfers, model execution, postprocessing, and tracking. Investigate one measured bottleneck. ONNX Runtime exposes profiling support for operator-level investigation. ([Profiling documentation](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html))

**Design question:** when is a faster model irrelevant because decode, transfers, or queuing dominate?

**Gate:** output/metric parity within predeclared tolerances; explicit active execution provider; no unnoticed CPU fallback; warm and cold measurements, p50/p95/p99, throughput, and memory tied to a complete hardware manifest. Follow the [measurement protocol](evaluation.md).

**Record:** `docs/modules/06-runtime.md`, export manifest, parity results, raw measurements, and the optimization decision.

## 07 — Add bounded streaming and a service

**Understand:** producers/consumers, bounded queues, backpressure, freshness, event time versus processing time, serialization, cancellation, and state ownership.

**Build:** retain an offline mode that processes every frame, and add a paced replay mode that simulates a source arriving at its original frame rate. One pipeline worker owns detector and tracker state. Add start/status/stop controls and server-sent events for metadata; use configured local sources initially. Reject a second active job explicitly.

**Explore:** slow inference, slow a result consumer, corrupt a frame, and stop/restart a job. Compare a bounded FIFO with latest-frame behavior. Observe frame age as well as throughput; a stable FPS can conceal stale results.

**Design question:** should overload preserve every frame or preserve freshness, and how can a client tell which guarantee it receives?

**Gate:** bounded input and result buffers, documented overflow policies, drop counts, ordered timestamps, recoverable errors, graceful shutdown, and no state leakage after restart. Test service lifecycle and overload with fake inference so CI does not require a GPU. Handle gaps by adapting tracker time steps or applying an explicit reset policy if the library assumes fixed cadence.

**Record:** `docs/modules/07-streaming.md`, API schema/examples, concurrency decisions, and overload experiment results.

## 08 — Package and operate the system

**Understand:** immutable artifacts, startup versus readiness, resource limits, telemetry cardinality, dependency compatibility, release promotion, and rollback.

**Build:** a Docker image for the chosen deployment target, startup configuration validation, health/readiness checks, structured logs, and a metrics endpoint. CI checks code/contracts and builds the image; heavyweight parity/performance checks run explicitly on the reference machine. Store artifacts by immutable version/hash and select the active version at startup.

**Explore:** missing model, wrong class metadata, unavailable accelerator, source failure, saturation, and process restart. Load the previous artifact/config to demonstrate rollback. Compare memory at startup and after a sustained replay.

**Design question:** which symptoms reveal a bad model, a bad source, or an overloaded runtime, and which ones should change readiness?

**Gate:** container smoke test; documented startup/recovery commands; a measured 30-minute replay at the target load with no unbounded memory or queue growth; visible latency/drop metrics; successful rollback drill. Local binding is the initial service default; remote exposure needs a separate deployment design.

**Record:** `docs/modules/08-operations.md`, runbook, metric definitions, release checklist, and one short incident postmortem.

## 09 — Analyze failures and publish a reproducible result

**Understand:** aggregate versus slice metrics, dataset shift, stress tests versus natural data, label uncertainty, and limits of offline evidence.

**Build:** a curated error gallery, small-object/occlusion/crowding slices, measured camera-motion examples, and lighting slices when labels or audited examples support them. Synthetic darkening/blur can probe robustness but must be labeled synthetic, not a night-scene benchmark. Choose one evidence-driven improvement; repeat affected accuracy and performance checks.

**Explore:** explain three consequential failures from input through intermediate outputs to root-cause hypothesis. Compare before/after with the same evaluation policy. Use final held-out data only after freezing choices; once inspected for tuning, it is no longer an untouched test set.

**Design question:** what evidence is still missing before using this in an autonomous system, and how would the interface expose uncertainty or degraded operation?

**Gate:** reproducible demo from a clean environment, measured model comparison, tracking report, runtime/service report, architecture diagram, data/model/system cards, tested runbook, and linked lessons. Cite limitations and missing hardware tests directly. No real-time claim without hardware, workload, freshness target, and measurements.

**Record:** `docs/modules/09-release.md`, final report, a short demo, and only a small evidence-backed backlog.

## Optional extensions — choose one after the core

| Extension | Learning question | Completion evidence |
| --- | --- | --- |
| TensorRT FP16 | How do engine compilation and target hardware affect deployment? | Build on target; record compatibility; compare parity and latency |
| INT8 quantization | How does calibration affect small-object recall? | Representative train-only calibration set; accuracy/performance tradeoff |
| RTSP/live camera | How do jitter, reconnects, and source timestamps affect freshness? | Reconnect drill, timestamp semantics, bounded buffers |
| Edge board or ROS 2 adapter | What changes under power/thermal or middleware constraints? | On-device sustained benchmark or explicit message/QoS contract |
| Drift investigation | Which observable signals justify collecting labels? | Simulated shift, signal analysis, manual review/retraining proposal |

Optional work does not block the core release. CPU-only development can complete pipeline, service, and ONNX work; obtain bounded accelerator time for meaningful fine-tuning if necessary. Do not silently replace the two-family training objective with an infeasible CPU schedule.

## References and tooling notes

Reviewed on 2026-09-29; recheck compatibility when pinning implementations.

- [VisDrone official dataset](https://github.com/VisDrone/VisDrone-Dataset): distinct DET and MOT tasks and task-specific downloads. Inspect terms and annotations for the chosen release.
- [VisDrone annotation documentation](https://github.com/zhaobaiyu/visdrone/blob/master/doc/dataset.md): ignored-region and category semantics must survive conversion.
- [RT-DETR official implementation](https://github.com/lyuwenyu/RT-DETR) and [original paper](https://arxiv.org/abs/2304.08069): architecture and implementation starting points; published benchmark numbers are not EdgeVision measurements.
- [Ultralytics export documentation](https://docs.ultralytics.com/modes/export/): export options depend on model/backend. Verify the actual selected combination.
- [Ultralytics licensing](https://www.ultralytics.com/license): review the chosen code and weights before settling the repository's licensing and distribution plan.
