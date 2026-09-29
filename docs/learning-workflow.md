# How we will learn and build

We work one module at a time. Each module connects an observable behavior to the mechanism behind it and then to a design decision. The guide should explain the concepts needed for the next step, help implement and debug it, and leave durable notes alongside the code.

## Module loop

1. **Frame the question.** State the module objective, prerequisite concepts, completion gate, and a small experiment budget. Connect it to the existing system.
2. **Predict.** Ask what we expect to happen and why: for example, whether higher input resolution will help small objects enough to justify latency. Predictions are useful even when wrong.
3. **Build the smallest working piece.** Keep the implementation readable. Reuse mature model/tracker internals while exposing the boundaries we need to understand.
4. **Inspect and perturb.** Trace a concrete input, examine intermediate representations, change one important variable, and inject a realistic failure.
5. **Measure and explain.** Compare results with the prediction. Distinguish observation, interpretation, and untested hypothesis. Run checks appropriate to changed behavior.
6. **Document and close.** Update module notes, run references, decisions, architecture/contracts if changed, and roadmap status. Explain remaining limitations and identify the next small step.

The learner can choose whether to implement a piece first or walk through it with the guide. Do not turn routine progress into a quiz or permission gate. Use occasional teach-back questions to reveal gaps: “Where do these coordinates live?” or “What happens if the consumer stops reading?”

## Documentation commitments

- Document important discoveries, mistakes, tradeoffs, and results in the same change as the relevant implementation whenever practical.
- Keep planned exercises separate from completed lessons. Never invent results, commands that were not run, or claims of reproducibility.
- Prefer a short explanation and one concrete worked example over transcripts or a copied textbook chapter.
- Record command, config, source revision, data/artifact identity, environment, and run ID for results we want to reproduce. Include only relevant log excerpts.
- Explain failures and why a fix worked. Negative findings and abandoned hypotheses are useful when they prevent repeated mistakes.
- Link to source code and primary references; avoid copying large upstream explanations.
- Update the architecture when the actual implementation diverges. Use an ADR for consequential choices, not every small refactor.

Use the [module template](templates/module.md). Each completed module should let a new reader answer: what was built, how it works internally, what was measured, what broke, why the design was chosen, and how to reproduce it.

## Scope controls

Keep one main experiment and one failure drill per module unless a gate exposes a real issue. Start with a tiny subset before expensive training. Cap training runs before launching them; run one deliberate ablation rather than an open-ended search. Add a tool only when its educational value or a concrete bottleneck pays for its setup and maintenance.

Record interesting tangents in a short backlog. If scope grows, remove or defer another item explicitly. The core release should remain usable without optional infrastructure or hardware.
