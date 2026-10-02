# 00 — Scope and budgets

Status: initial planning gate complete. Targets below are provisional design hypotheses, not measured service guarantees. Hardware inventory was inspected on 2026-09-29; learner availability came from the user.

## Operating scenario

Analyze one recorded aerial urban-road video for people and road users. Begin with offline processing that preserves every frame. Later, paced replay will test whether a bounded queue can maintain fresh results under load. No flight control or live-network source is required.

The first technical reference clip is eight repeats of the packaged Ultralytics bus image, encoded at 10 FPS and 540×720. It exercises video plumbing and visible detections. It is not aerial footage or an evaluation dataset. Module 02 will select a licensed, documented aerial reference sequence and class mapping. Module 01 retains all COCO-80 outputs; the target product's people/vehicle/bicycle classes are not yet dataset labels.

## Environment and budgets

| Item | Initial decision / observation |
| --- | --- |
| Machine | Minisforum V3, Ryzen 7 8840U / Radeon 780M, 32 GB physical RAM (user-provided) |
| Development environment | Ubuntu 24.04.4 under WSL2, x86-64, Linux 6.6.87.2 |
| CPU capacity | 8 cores / 16 logical threads visible; start inference with 4 PyTorch threads |
| WSL resources | Approximately 25 GiB visible RAM, 764 GiB available filesystem space at inspection |
| Python/tooling | System Python 3.12.3, uv 0.12.12; local locked virtual environment |
| Learning time | Up to 10 hours/week; roadmap estimate 9–13 active weeks plus contingency |
| Reference runtime | CPU PyTorch initially; CPU ONNX in module 06 |
| Initial input | Batch 1, model input 320×320; preserve source aspect ratio with padding |
| Future representative workload | One approximately 1280×720 aerial clip; paced replay at 5 input frames/s initially |
| Provisional throughput target | Sustain 5 completed frames/s at the declared workload |
| Provisional freshness target | p95 scheduled-arrival-to-result ≤500 ms in paced replay, after startup |
| Provisional process memory ceiling | 4 GiB resident memory; measure before treating this as an acceptance limit |
| Initial training budget | No full training runs now; tiny correctness exercises only |
| Cloud budget | $0 for this milestone; no provider or paid session selected |

**Why begin at 5 FPS?** This is a deliberately modest *experiment workload* for one stream on a CPU-first notebook, not a requirement derived from drone operations. Five arrivals per second give an arrival interval of `1 / 5 s = 0.2 s = 200 ms`. That is slow enough to investigate decoding, detection, timing, and overload on the local machine while still exposing whether the pipeline keeps up with a continuing source. At 30 source FPS, observing only 5 FPS would mean explicitly selecting one in every six frames; that sampling would need to be reported, and it could harm tracking. Offline evaluation will still preserve every frame. The chosen rate, aerial resolution, and model input size form one workload definition; changing any of them calls for new measurements.

The eight-frame smoke run cannot validate sustained throughput, freshness, or memory limits. Its later detector calls took 49.85–83.24 ms (61.39 ms median), but `detector_ms` includes preprocessing and postprocessing, uses a repeated non-aerial image, and excludes output writing. It does **not** measure model-forward time or sustained 5 FPS. See the [smoke report](../reports/module01-smoke.md). A miss will trigger profiling and an explicit requirement/workload revision rather than hidden frame dropping.

## Provisional per-frame stage budget

At 5 arrivals/s, spending the entire **200 ms** arrival interval on each frame leaves no room for variation. The first draft did exactly that (`10 + 10 + 150 + 10 + 20 = 200 ms`), so it was an unsound budget for *sustained* 5 FPS. The revised planning target is **150 ms of complete processing per frame**. All stage numbers below are allocations to test, **not measured per-stage timings**.

| Stage | Planning allocation | Why start here? |
| --- | ---: | --- |
| Decode 1280×720 video to RGB | 10 ms | Placeholder for a single local stream; source codec and I/O may change it. |
| Resize/pad/tensor conversion to 320×320 | 10 ms | Simple CPU operation, but copy/interpolation cost still needs measurement. |
| Model forward on CPU | 100 ms | The largest allowance and a design target for a small detector. The smoke run's 61.39 ms median covered **more** than forward alone, but too few unrepresentative frames to establish this bound. |
| Confidence filtering, NMS, and future tracking | 10 ms | Placeholder; crowded scenes and tracker cost can exceed it. |
| Serialize results and write output | 20 ms | Allows for JSON and optional image overlays; disk and image encoding can vary. |
| **Complete sequential processing** | **150 ms** | **50 ms left per 200 ms arrival interval.** |

The arithmetic is `10 + 10 + 100 + 10 + 20 = 150 ms`. At that *hypothetical constant* service time, one worker could process `1000 / 150 ≈ 6.67` frames/s; a 5 FPS source would use `150 / 200 = 75%` of its time, leaving `50 / 200 = 25%` as headroom. This is a capacity sketch, not a throughput prediction: startup, slow frames, queueing, operating-system contention, and variation can still cause backlog. Sustained throughput requires measured average service time below 200 ms **and** acceptable tail latency and drop rate on the declared workload.

**Why 500 ms p95 freshness?** At 5 FPS, it is `500 / 200 = 2.5` frame-arrival intervals. This is a provisional tolerance for occasional delay during a learning/demo pipeline, not an autonomous-system safety limit. If a frame uses its 150 ms processing allowance, it has `500 - 150 = 350 ms` left for waiting and scheduling before missing the target. For illustration, with constant 150 ms service and two complete frames ahead of it, age would be approximately `2 × 150 + 150 = 450 ms`; with three ahead, about `3 × 150 + 150 = 600 ms`. Thus queue depth and overload policy matter even when the average processing rate exceeds the arrival rate. These examples assume no extra source delay; they do **not** prove p95 will be below 500 ms.

Measure freshness from **scheduled source arrival to completed result using a common monotonic host clock**, after startup. The p95 target means at least 95% of *completed* frames in the declared test should meet 500 ms; report dropped frames separately so dropping slow work cannot make latency look good. The paired 5-completed-FPS target at 5 offered FPS also requires keeping up without sustained drops. Module 07 will implement paced replay and check both targets on a representative clip; the final queue capacity and targets may then change. A model with a 25 ms forward pass alone does not guarantee 40 end-to-end FPS because every other stage and any queue wait also take time.

## Compute portability

The Radeon 780M is not a CUDA target. Other integrated-GPU backends are optional experiments; TensorRT is outside the core local scope. Use local CPU for data validation, inference, tests, and service work, and consider external GPU sessions for meaningful fine-tuning later.

Before any paid training, establish maximum total spend and run duration. On interruptible compute, validate save/resume and persist checkpoints, optimizer/scheduler/progress state, RNG state where supported, resolved config, and data identity outside the ephemeral instance. Stop compute when done and account for retained storage. No cloud resources have been provisioned.

## Lessons and open questions

Training compute and deployment compute are separate decisions. WSL-visible memory differs from physical machine RAM, and latency budgets belong to the whole pipeline. The immediate open questions are representative aerial workload selection and measured local performance; neither blocks learning the first inference contract.
