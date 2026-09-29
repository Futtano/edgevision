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

- Exact detector checkpoints, framework versions, licensing, and compute budget — modules 00–03.
- Dataset conversion/ignore policy and held-out evaluation protocol — module 02.
- Deployment model choice based on accuracy, latency, and memory — module 04.
- Runtime precision and export compatibility — module 06.
- Queue policy, frame-gap behavior, and service lifecycle — module 07.
- Promotion/rollback procedure and operating targets — module 08.
