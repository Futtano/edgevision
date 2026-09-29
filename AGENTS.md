# Working on EdgeVision

This repository is both a production computer-vision learning project and reusable learning material. Read `README.md`, `docs/roadmap.md`, and `docs/learning-workflow.md` before substantial implementation. Follow applicable module contracts and completion gates; explicit user instructions take precedence.

- Progress in small, explainable modules. Keep the core scope to one machine/stream, two detector families, and one tracker unless the user changes it.
- Explain relevant internals and design tradeoffs alongside implementation. Use a concrete frame, tensor, timing trace, or state transition when helpful.
- Maintain notes in `docs/modules/` using `docs/templates/module.md`. Record important discoveries, failures, evidence, and reproducible commands with the change. Update the module index honestly.
- Distinguish plans, measured observations, and hypotheses. Do not invent benchmark numbers, successful checks, or completed learning outcomes.
- Maintain architecture/contracts and record consequential choices in `docs/decisions/`. Keep implementation simpler than the optional extension backlog.
- Use meaningful boundary, data, and lifecycle checks. CPU CI should not require dataset downloads, model downloads, or a GPU. Run heavier model and runtime checks explicitly when relevant.
- Preserve dataset provenance, class/ignore semantics, and split integrity. Keep large datasets, video, weights, caches, and experiment stores out of Git.
- Do not launch expensive training or add paid infrastructure without an established compute budget. First run a bounded smoke check.

The initial repository contains planning documentation only; planned paths and dependencies are not yet implemented or installed.
