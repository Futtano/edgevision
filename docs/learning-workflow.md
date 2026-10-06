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

## Local checks before commits and pushes

Install the locked environment and Git hooks once per checkout:

```bash
uv sync --locked --extra inference --python 3.12
.venv/bin/pre-commit install
```

For base-only CPU development, omit `--extra inference`. Hook installation enables both `pre-commit` and `pre-push` using the defaults in [.pre-commit-config.yaml](../.pre-commit-config.yaml). It is local to your checkout; committing the configuration does not install hooks for other contributors.

Both stages run Ruff lint, Ruff formatting checks, and ty with the project environment's locked tools. Checks cover the whole project for Ruff and the configured application-source scope for ty, even when only documentation changes. Formatting is checked without rewriting files; correct formatting with `.venv/bin/ruff format .`, review the changes, and stage them before retrying.

Run the hooks manually before committing:

```bash
.venv/bin/pre-commit run --all-files
.venv/bin/pre-commit run --all-files --hook-stage pre-push
```

Local hook entries use `uv run --locked --no-sync` to retain optional inference packages and avoid changing the environment during Git operations. Synchronize explicitly after dependency or lockfile changes. CI creates the locked base environment and runs the same hook configuration, followed by pytest. Tests remain a separate check; run `.venv/bin/pytest -q` locally when validating a change. No model or dataset downloads are required by these hooks. See the [pre-commit reference](https://pre-commit.com/) for hook lifecycle details.

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
