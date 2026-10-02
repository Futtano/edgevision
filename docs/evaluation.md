# Evaluation and evidence protocol

This is a plan. No scores, speed claims, or acceptance tolerances have been measured yet. Fill in hardware-specific targets in module 00 and freeze export/release thresholds before examining candidate results.

## Data and accuracy

Preserve official train/validation/test boundaries. Use training data for fitting and any calibration, validation for model/threshold selection, and an accessible labeled held-out split for the final report. Record sequence overlap and provenance when combining detection and tracking tasks; do not randomly split neighboring frames. If a clean held-out set is unavailable, report that limitation instead of relabeling validation results as test performance.

Maintain a class-map version and original annotations, including ignored regions. Define filtering, maximum detections, confidence policy, and matching behavior for every metric. A convenience format converter that discards ignored labels does not automatically implement official evaluation. See the [VisDrone annotation specification](https://github.com/zhaobaiyu/visdrone/blob/master/doc/dataset.md).

| Question | Evidence |
| --- | --- |
| Does the detector localize and classify well? | AP50:95, AP50, per-class AP, recall; named evaluator/version and settings |
| Which objects fail? | Size and occlusion slices with explicit definitions, counts, and example errors |
| Does identity persist? | HOTA, IDF1, ID switches, per-sequence results; evaluator preprocessing and class policy |
| Are two models comparable? | Same splits, evaluator, hardware; disclosed resolution, pretraining, tuning and compute budgets |
| Did export change behavior? | Box/score comparisons on matched predictions plus task metrics over validation data |

For AP, use an evaluation confidence policy that retains the precision/recall curve; do not blindly reuse the display threshold. Tracking may require lower-score detections than a visual demo displays. Keep these thresholds distinct and recorded.

Use complete ordered sequences for benchmark MOT results. A streamed run with dropped frames is a separate system experiment. Select or validate a VisDrone-compatible adapter before relying on [TrackEval](https://github.com/JonathonLuiten/TrackEval); metric availability does not guarantee dataset-format compatibility.

Never compare post-NMS boxes solely by array index: prediction order may change across runtimes. Define matching, absolute/relative numeric tolerances, and permitted metric deltas before parity testing. Record tolerances in config; failures require investigation, not retrospective loosening without explanation.

## Runtime measurement

Every report includes CPU, GPU/VRAM, RAM, OS, relevant driver/CUDA/runtime versions, active execution provider, model and artifact hashes, precision, input resolution, batch size, source clip, decode backend, threads, concurrency, warmup, and power mode when available.

Separate the following measurements:

1. **Cold startup:** process launch to usable model and first completed result; specify whether download/engine compilation is included.
2. **Model execution:** forward pass with its exact input/output boundary. Synchronize asynchronous GPU work correctly or use device events; report transfer costs separately if excluded.
3. **Processing latency:** decode/preprocess/transfer/infer/postprocess/track/serialize stages and complete per-frame processing time.
4. **Queue wait and frame age:** ingest to inference start, and ingest to result completion on one monotonic clock. For paced replay, also measure scheduled arrival to completion. Real capture-to-result latency requires trustworthy capture timestamps and clock alignment.
5. **Throughput and loss:** decoded, processed, emitted, and dropped frames per interval, with drop reasons. Report offered input rate and useful completed FPS separately.
6. **Resources:** peak process memory, accelerator allocated/reserved memory where relevant, device memory/utilization from a named sampling source, and queue depths. Unavailable GPU metrics are marked unavailable.

Starting benchmark protocol: batch size 1, one stream, at least 50 warmup frames, then at least 1,000 measured frames; repeat three times under the same conditions. Extend warmup if traces show instability. Save individual timing samples, summarize each run and run-to-run spread, and note that p99 from short runs is noisy. Do not equate reciprocal median forward latency with complete pipeline throughput.

Run offline throughput and paced replay separately. Use a fixed reference clip and include crowded frames because postprocessing and tracking cost depend on object counts. Run at the target source rate and a deliberately overloaded rate. The module 08 soak is a separate 30-minute check for resource growth and recovery behavior.

## Acceptance targets to fill in

| Target | Decision point | Evidence needed |
| --- | --- | --- |
| Source rate/resolution, runtime hardware | Module 00 | Hardware inventory and scenario |
| Minimum useful FPS, maximum p95 frame age | Provisional in 00; revisit after 01; freeze before 07 paced replay | Representative staged timing and paced replay report |
| Memory ceiling and queue capacity | Provisional in 00; refine in 07 | Peak/steady-state memory and overload drill |
| Export AP/recall and numerical tolerance | Before module 06 comparison | Parity report |
| Permitted release accuracy regression | Before candidate promotion | Same-protocol baseline/candidate results |
| Startup/recovery expectations | Before module 08 drills | Timed startup, failure and rollback evidence |

A miss is a useful finding: profile it, reduce the workload or revise the requirement explicitly, and record the tradeoff. Do not invent universal AP or latency thresholds before there is a baseline.

## Test and release boundaries

CPU CI checks coordinate transforms, class/annotation validation, known metric examples, configuration errors, tracker lifecycle via fixtures, service lifecycle, and overload behavior. Keep fixtures synthetic or redistribution-safe. Real model loading/export and GPU performance belong to explicit integration checks on the reference machine, not a flaky mandatory hosted-CI benchmark.

A candidate release requires traceable data/model/config versions, completed accuracy and runtime comparisons, a container smoke test, and a rollback drill. Promotion selects an immutable model bundle plus compatible config; retain the previous bundle. A release report links exact commands and raw results, identifies open limitations, and never substitutes an attractive overlay for quantitative evidence.
