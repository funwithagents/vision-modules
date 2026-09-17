# Vision Modules

Vision Modules is a library of independent vision modules that run over a video stream and turn it into perception a multimodal robot, system, or agent can act on. The core idea is that perception is modular: each module solves one perceptual job over the same incoming stream and exposes it behind a uniform interface, so richer perception means enabling another module rather than rewriting the pipeline. Developers depend on this library to add vision capabilities to their own robot, system, or agent, switching on only the modules their use case needs. Every module is designed as its own spec here before it is built.

## Specs

<!-- One row per concept spec. Keep the Status column in sync with each spec's `**Status:**` line. -->

| Spec | Description | Status |
|---|---|---|
| [project.md](project.md) | Project structure and tooling: Python version, packaging with uv, layout conventions, feature-named dependency extras | Implemented |
| [testing.md](testing.md) | Testing strategy: two-tier `tests/`/`tests-e2e/` split, functional-test philosophy, skip-without-credentials live tier | Implemented |
| [stream.md](stream.md) | Stream: single capture entry point (`StreamProvider`), `Frame` shape, frame-ownership rules, injectable `FrameSource`, measured vs. nominal fps | Implemented |
| [pipeline.md](pipeline.md) | Pipeline: staged-graph runtime — latest-value sampling, `Stage`/`Module` base classes, `last_input`, `Result` contract, lifecycle and threading rules | Implemented |
| [hand.md](hand.md) | Hand: shared hand stage (`HandStage`/`HandResult`) detecting and cropping hands once per frame behind a boxes-only `HandDetector` seam | Implemented |
| [gesture_classifier.md](gesture_classifier.md) | Gesture classifier module: named gestures from the hand crop via an `ImageClassifier` seam (HaGRID ViT shipped) | Implemented |
| [hand_demo.md](hand_demo.md) | Hand demo: the `examples/hand_demo.py` Gradio browser app wiring the pipeline end-to-end, with per-module snapshot buttons | Implemented |
| [snapshot.md](snapshot.md) | Snapshot: `save_frame` / `save_crops` / `save_input` — caller-side functions writing what a node last published or last consumed to a local image file | Implemented |

Each spec also opens with a YAML **frontmatter** block declaring the `code:` and `tests:` files it governs — the spec → code/tests mapping the spec-drift checks use to scope what they compare. Keep it current when files move, and see [AGENTS.md](../AGENTS.md) ("Spec frontmatter") for the full convention.

### Status legend

- **Not started** — no design decisions made yet
- **Draft** — actively being brainstormed/defined, contains open questions
- **Stable** — design settled, reviewed and validated (open questions are deferrals only), **ready to implement but not necessarily implemented yet**. This is the design-review gate, before code is written.
- **Implemented** — a **Stable** spec that a `Done` plan has built: the code now exists and matches the spec (design and code in sync)
- **Updated** — an **Implemented** spec since edited in a way that needs new code, so the code no longer matches it; a new implementation plan is needed (or in progress) to catch up. Returns to **Implemented** once that plan is `Done`.
