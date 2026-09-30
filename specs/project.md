---
code:
  - pyproject.toml
tests:
  - tests/test_project_map.py
  - tests/test_core_imports.py
---

# Project

**Status:** Implemented

## Purpose

Structure and tooling for the Vision Modules project itself: Python version, dependency/packaging management with `uv`, repo layout conventions, and development tooling.

## Decided

- **Python version:** 3.12+ minimum.
- **License:** MIT (`LICENSE` at the repo root), declared in `pyproject.toml` via `license = "MIT"` and `license-files = ["LICENSE"]` so built distributions carry it. Models downloaded at runtime are not part of the repo and keep their own licenses (see the README's License section).
- **Package layout:** `src/` layout — `src/vision_modules/...` — not flat, to avoid accidentally importing an uninstalled package from the repo root.
- **Dependency/venv management:** `uv`. Dev tooling lives in the `dev` dependency group (`uv sync --dev`), not in runtime `dependencies`.
- **Runtime dependencies are split by feature; the core depends on `numpy` alone.** Core `dependencies` hold only `numpy` (the `Frame` / result arrays). That is everything needed to run the stream and pipeline runtime ([stream.md](stream.md), [pipeline.md](pipeline.md)) over a feed the caller provides through a `FrameSource`, and to write custom stages. Every device, codec and model goes in an **optional extra named after the feature**, under `[project.optional-dependencies]`, so a user installs only what they enable:
  - `opencv` = `opencv-python`: opening a webcam, file or URL through `OpenCVSource` (so `StreamProvider(0)` / `StreamProvider("clip.mp4")`, [stream.md](stream.md)) and encoding snapshots ([snapshot.md](snapshot.md)). Any OpenCV wheel variant (`-headless`, `-contrib`, …) provides the same `cv2` module and works just as well. A consumer that already has one can skip the extra. The extra names `opencv-python` only as the default for everyone else.
  - `hand` = `mediapipe` (the shipped hand detector, [hand.md](hand.md)) + `torch`, `transformers`, `pillow` (the shipped gesture classifier, [gesture_classifier.md](gesture_classifier.md)). `mediapipe` brings its own `opencv-contrib-python`, so `[opencv,hand]` installs two `cv2` variants side by side. This is accepted: they provide the same module, and it was already the case when `opencv-python` was a core dependency.

  - `face-mediapipe` = `mediapipe`: the shipped face detector ([face.md](face.md)). The face family names its extras per backend (`face-<backend>`), because its backends will have unrelated dependencies (MediaPipe today, OpenCV Zoo or torch for a recogniser later) and a consumer installs only the one it runs. Same `mediapipe` range as `hand`, and the same `opencv-contrib-python` side by side note applies.

  `uv add --optional <extra> …` / `pip install "vision-modules[opencv,hand]"`. A new feature family (faces, poses, …) gets its own extra; the `dev` group installs every extra so the full test suite and the demo can import everything.
- **The core imports no optional package at module level.** Importing `vision_modules` (or any of its modules) must succeed with only `numpy` installed. An optional package is imported inside the function or method that needs it (`mediapipe` and `torch`/`transformers`/`PIL` already are), and `cv2` is imported only where OpenCV is actually needed: `OpenCVSource.open()` and the snapshot writer. When it is missing, the `ImportError` names the extra (`pip install "vision-modules[opencv]"`). Everything else that only needs array operations (the `mirror` flip, BGR→RGB for a model) uses numpy. The root keeps re-exporting every public name ([snapshot.md](snapshot.md), [stream.md](stream.md), …); since no import is eager, that costs a numpy-only user nothing. `tests/test_core_imports.py` pins this: in a subprocess where `cv2`, `mediapipe`, `torch`, `transformers` and `PIL` are blocked, it imports the package, runs a `StreamProvider` over an injected source through a `Stage`, and checks that the OpenCV-backed paths fail with the extra's name and `MediaPipeFaceDetector()` with `face-mediapipe`'s.
- **Dev/example-only tooling is a `[dependency-groups]` entry, never a package extra.** A dependency that no library consumer would ever install — because it belongs to something in this repo, not to the library ([hand_demo.md](hand_demo.md)'s `gradio`, for instance) — goes in its own named group under `[dependency-groups]` (PEP 735), pulled into `dev` with `{include-group = "…"}`. This keeps `pip install "vision-modules[...]"` free of dependencies no consumer asked for, while `uv sync --dev` still installs everything needed to run the repo's own scripts and tests.
- **Linting/formatting:** `ruff`.
- **Testing:** `pytest`, in two physically-separated tiers — a fast, deterministic, no-network default run (`tests/`, the only tier `testpaths` collects) and an opt-in live tier (`tests-e2e/`) that calls real external services. Full strategy is specced in [testing.md](testing.md).
- **Type checking:** `pyright` (`standard` mode), a dev dependency run via `uv run pyright`. Config lives in `[tool.pyright]` in `pyproject.toml`, targeting `src`, `tests`, `tests-e2e` and `examples` (the demo is spec-governed code and goes through the same gate), pinned to the `.venv`. `examples` is also on pyright's `extraPaths` and pytest's `pythonpath` so the fast tier can import the demo module's helpers.
- **Repo shape:**
  - `src/vision_modules/` — the package, one module per core concept.
  - `specs/` — pre-implementation design docs, one per concept (this folder).
  - `plans/` — implementation plans turning settled specs into buildable steps.
  - `tests/` at repo root, mirroring the `src/vision_modules/` module structure.
  - `tests-e2e/` at repo root, for the live tier above — not collected by the default `pytest` run.
  - `examples/` at repo root — runnable applications built on the library (not part of the package), type-checked with it.

## Open questions

None currently.
