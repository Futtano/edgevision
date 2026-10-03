# 0003 — Preserve native VisDrone annotations

Date: 2026-10-03

Status: accepted for Module 02's first data step. Training export and official evaluation behavior remain unimplemented.

**Context:** pretrained COCO detections and VisDrone ground truth have different label spaces and metadata. Converting immediately to a training label format could discard ignore regions and attributes before evaluation semantics are understood.

**Decision:** parse DET ground truth into a separate immutable annotation record, retaining native category, score, box, truncation, occlusion, and source line. Derive corner coordinates and the versioned `visdrone-det-10-v1` class mapping. Preserve excluded rows with explicit roles. Report out-of-bounds and duplicate rows without modifying source data. Begin with one-image auditing and synthetic fixtures; establish dataset and evaluator contracts in subsequent steps.

**Alternatives:** exporting only class/normalized-box training labels now would simplify a training loader but remove evaluation-relevant information. Reusing inference `Detection` would conflate inclusion flags with confidence and native annotations with predictions.

**Consequences:** more metadata is retained than a detector trains on. Reports can be traced to source content and audited visually. No official-score claim follows from merely preserving ignore information. The current caller-declared split is metadata, not proof of split integrity. Model inference remains COCO-80 until the later training integration.

**Revisit when:** real-data inspection exposes a format variation, the training adapter requires a derived format, or reference-evaluator parity establishes a more precise conversion/ignore policy. Preserve raw annotations through those changes.
