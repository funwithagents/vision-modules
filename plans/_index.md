# Implementation plans

Implementation plans for Vision Modules — each plan turns a settled part of a spec (see [specs/_index.md](../specs/_index.md)) into concrete, buildable steps. Plans are ordered by their date-time filename prefix (`YYYYMMDDHHmm_`).

## Plans

<!-- One row per plan, chronological by filename prefix. Keep the Status column in sync with each plan's `**Status:**` line. -->

| Plan | Description | Status |
|---|---|---|
| [202609161500_dependency-extras.md](202609161500_dependency-extras.md) | Declare core runtime deps (`numpy`, `opencv-python`) and the feature-named `hand` extra in `pyproject.toml` | Done |
| [202609161510_pipeline-runtime.md](202609161510_pipeline-runtime.md) | `pipeline.py`: `LatestValue`, `Result`, `Stage`/`Module` worker loop, `Pipeline` lifecycle, with tests | Done |
| [202609161520_stream-provider.md](202609161520_stream-provider.md) | `stream.py`: `Frame`, `FrameSource`/`OpenCVSource`, `StreamProvider` capture thread, fast tests + camera live test | Done |
| [202609161530_hand-stage.md](202609161530_hand-stage.md) | `hand.py`: `Hand`/`HandResult`, `padded_box`, `HandStage`, MediaPipe Tasks detector with model download, tests | Done |
| [202609161540_gesture-classifier.md](202609161540_gesture-classifier.md) | `gesture_classifier.py`: `Gesture`, `GestureClassifier` module, HaGRID ViT backend, `select_device`, tests | Done |
| [202609161550_hand-demo.md](202609161550_hand-demo.md) | Public re-exports in `__init__.py`, `examples/hand_demo.py` Gradio webcam demo, import-cost test | Done |
| [202609171400_stage-published-count.md](202609171400_stage-published-count.md) | `Stage.published_count`: fixes the demo's fps meter, which was measuring upstream `frame_id` drift instead of the stage's own rate | Done |
| [202609171600_restartable-nodes.md](202609171600_restartable-nodes.md) | Restartable nodes (`FrameSource.open()`, owned vs borrowed backends), `Pipeline.start()` rollback, stream close-then-join with a locked `OpenCVSource`, `target_fps` validation, public `DetectedHand`/`Upstream`, demo tests | Done |
| [202609171700_stream-fps.md](202609171700_stream-fps.md) | Stream fps: `StreamProvider.published_count` (measured, same contract as `Stage`) and `source_fps` / `FrameSource.fps()` (nominal, `CAP_PROP_FPS`); protocol doubles in tests and demo | Done |
| [202609170227_snapshot.md](202609170227_snapshot.md) | `snapshot.py`: `save_frame` / `save_crops` over published values, package exports, round-trip and pipeline-driven tests | Done |
| [202609170236_save-input.md](202609170236_save-input.md) | `Stage.last_input` (the item last handed to `process()`) and `save_input(stage, path)` dispatching on it, so a label-only module's input crop can be saved | Done |
| [202609171209_demo-snapshot-buttons.md](202609171209_demo-snapshot-buttons.md) | Hand demo: snapshot folder box + "Save detector/classifier snapshot" buttons writing each module's last input as `snapshot_<stage>_<timestamp>.jpg`, with tested helpers | Done |
| [202609171800_multi-hand-selection.md](202609171800_multi-hand-selection.md) | Multi-hand: `HandStage` `max_hands` default 2 (detected and published), area rank with hysteresis, slot-stable order via `select_hands`; demo `--max-hands` and one colour-linked panel per hand | Done |

## Status legend

- **Todo** — written, not yet started
- **In progress** — actively being implemented
- **Done** — implemented, verified (lint/type-check/tests pass), and merged
