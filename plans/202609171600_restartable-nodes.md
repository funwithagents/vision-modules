# Restartable nodes and lifecycle fixes

**Status:** Done

Implements the settled behavior in `specs/pipeline.md` ("Lifecycle", "`Stage` — the generic node") and `specs/stream.md` ("API", "Capture loop contract"), plus the ownership rules those changes forced into `specs/hand.md` and `specs/gesture_classifier.md`. Delivers: every node can be stopped and started again; `Pipeline.start()` rolls back on failure; the stream's stop order is the one the code actually needs (close, then join) with `OpenCVSource` made safe under it; `target_fps` rejects non-positive values instead of silently killing the worker; `DetectedHand`/`Upstream` become public; `examples/` joins the type-check gate and the demo's pure helpers get tests. Deliberately leaves out: a push/callback delivery path and any change to the latest-value model.

## Scope

- `src/vision_modules/pipeline.py` — `target_fps` becomes a validated property (`ValueError` when not `None` and `<= 0`); `_run()` reads it once per iteration; `Pipeline.start()` stops already-started nodes (reverse order) when a later `start()` raises, then re-raises.
- `src/vision_modules/stream.py` — `FrameSource` gains `open()`; `OpenCVSource` opens in `open()` (not `__init__`), guards `read()`/`close()` with a lock so `release()` never runs concurrently with a read; `StreamProvider.start()` opens the source (so a restart re-opens it) and resets `ended`; `stop()` closes then joins; the worker only sets `ended` when the source ran out on its own, not when `stop()` interrupted it.
- `src/vision_modules/hand.py` — `HandStage` distinguishes the detector it created (owned: closed in `close()` and recreated on the next run) from an injected one (borrowed: never closed).
- `src/vision_modules/gesture_classifier.py` — same owned/borrowed split for the classifier.
- `src/vision_modules/__init__.py` — export `DetectedHand` and `Upstream`.
- `examples/hand_demo.py` — `PushFrameSource.open()`; `gradio` imported inside `main()` so the module is importable by the fast test tier without paying for the gradio import.
- `pyproject.toml` — pyright includes `examples` (with `extraPaths` so tests can import the demo module); pytest `pythonpath = ["examples"]`.
- `tests/test_pipeline.py` — restart, `target_fps` validation, `Pipeline.start()` rollback.
- `tests/test_stream.py` — `open()` on the scripted source, restart, `ended` stays `False` after an interrupted stop, real `OpenCVSource` round-trip on a generated video file.
- `tests/test_hand.py`, `tests/test_gesture_classifier.py` — borrowed backend survives restart and is not closed; owned backend (a fake substituted for the MediaPipe/HaGRID class) is created lazily, closed on stop, recreated on restart.
- `tests/test_hand_demo.py` — new: `FpsMeter`, `summarize`, `PushFrameSource`.
- `README.md` — stop order, `FrameSource` snippet, restart, new exports, `ruff format` fix.
- Specs: `pipeline.md`, `stream.md`, `hand.md`, `gesture_classifier.md`, `hand_demo.md`, `project.md`, `testing.md` — updated in the same change (see each spec's Decided section); `pipeline.md` and `stream.md` went through `Updated` and return to `Implemented` when this plan is `Done`.

## Steps

1. Mark this plan `In progress` (file + index).
2. Spec edits (all of the above), with `pipeline.md`/`stream.md` at `Updated` until step 6.
3. `pipeline.py`, `stream.py`, `hand.py`, `gesture_classifier.py`, `__init__.py` as scoped.
4. `examples/hand_demo.py` and `pyproject.toml`.
5. Tests as scoped; `README.md`.
6. Verification gate; flip `pipeline.md`/`stream.md` back to `Implemented` (file + index) and this plan to `Done` (file + index).

## Verification

```
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv run python examples/hand_demo.py --help
```
