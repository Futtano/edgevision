# EdgeVision

A learning project in production computer vision: detect and track objects in aerial video, then measure and improve the complete inference system.

The goal is to develop deep-learning engineering and system-design skills through a small, explainable implementation. Each module produces working software, reproducible evidence, and notes explaining its internals and tradeoffs.

**Status:** the first CPU video-inference slice is implemented and smoke-tested. No model has been trained, aerial accuracy evaluated, or streaming service implemented.

## Start here

- [Project roadmap](docs/roadmap.md): scope, sequence, exercises, and completion gates.
- [System architecture](docs/architecture.md): components, contracts, and runtime behavior.
- [Learning workflow](docs/learning-workflow.md): how we will work and record lessons.
- [Evaluation protocol](docs/evaluation.md): accuracy, tracking, performance, and release evidence.
- [Decision log](docs/decisions/README.md): design choices and their rationale.
- [Module notes](docs/modules/README.md): the evolving learning material.

## Intended result

A reproducible single-machine pipeline that turns aerial video into timestamped detections and object tracks. It will compare a compact convolutional detector with an RT-DETR-family detector, support PyTorch and ONNX inference, expose results through a small service, and include a measured release report and operational runbook.

The core project uses recorded video, one active stream, one tracker, and one deployment target. TensorRT, INT8, live cameras, and additional infrastructure are optional extensions. Performance targets will be set against the available hardware rather than assumed from model marketing benchmarks.

This is a perception engineering project. Flight control, navigation, sensor fusion, and safety validation are outside its scope.

## Begin the first module

Follow the [frame-by-frame learning walkthrough](docs/modules/01-inference.md) for setup, a runnable demo, a coordinate-transform exercise, and the actual model tensor trace. See the [smoke report](docs/reports/module01-smoke.md) for measured observations and limitations.

The prepared workspace has a model and demo clip. Run a new inference session with:

```bash
.venv/bin/edgevision infer --config configs/inference.yaml \
  --output-dir artifacts/runs/my-first-run
```

Each run writes JSONL detections, original-frame overlays, resolved configuration, provenance, and a completion/failure summary. Choose a new output directory each time. For a fresh checkout, follow the setup instructions in the walkthrough first.

Our [scope notes](docs/modules/00-scope.md) record the CPU deployment target, development environment, provisional operating budgets, and availability of up to 10 learning hours/week.
