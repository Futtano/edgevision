# EdgeVision

A learning project in production computer vision: detect and track objects in aerial video, then measure and improve the complete inference system.

The goal is to develop deep-learning engineering and system-design skills through a small, explainable implementation. Each module produces working software, reproducible evidence, and notes explaining its internals and tradeoffs.

**Status:** planning only. No models have been trained, benchmarks measured, or service implemented.

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

## First milestone

Module 00 is in progress: the [scope notes](docs/modules/00-scope.md) record your Ryzen 7 8840U / Radeon 780M machine, 32 GB RAM, and availability of up to 10 hours/week. Local deployment starts on CPU, with possible external GPU sessions for fine-tuning. The operating scenario and success criteria remain to be defined. Module 01 then delivers the first thin vertical slice: a short local video through a pretrained detector to inspectable JSON results. See the [roadmap](docs/roadmap.md) before starting implementation.
