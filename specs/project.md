---
code:
  - pyproject.toml
tests:
  - tests/test_project_map.py
---

# Project

**Status:** Implemented

## Purpose

Structure and tooling for the Vision Modules project itself: Python version, dependency/packaging management with `uv`, repo layout conventions, and development tooling.

## Decided

- **Python version:** 3.12+ minimum.
- **Package layout:** `src/` layout — `src/vision_modules/...` — not flat, to avoid accidentally importing an uninstalled package from the repo root.
- **Dependency/venv management:** `uv`. Dev tooling lives in the `dev` dependency group (`uv sync --dev`), not in runtime `dependencies`.
- **Runtime dependencies are split by feature.** Core `dependencies` hold only what every user needs: `numpy` (the `Frame` / result arrays) and `opencv-python` (capture and image ops, see [stream.md](stream.md)). Heavy, feature-specific packages go in an **optional extra named after the feature**, under `[project.optional-dependencies]`, so a user installs only the perception they enable. Today there is one: `hand` = `mediapipe` (the shared hand stage, [hand.md](hand.md)) + `torch`, `transformers`, `pillow` (the gesture classifier, [gesture_classifier.md](gesture_classifier.md)) — `uv add --optional hand …` / `pip install "vision-modules[hand]"`. A new feature family (faces, poses, …) gets its own extra; the `dev` group installs every extra so the full test suite can import everything.
- **Linting/formatting:** `ruff`.
- **Testing:** `pytest`, in two physically-separated tiers — a fast, deterministic, no-network default run (`tests/`, the only tier `testpaths` collects) and an opt-in live tier (`tests-e2e/`) that calls real external services. Full strategy is specced in [testing.md](testing.md).
- **Type checking:** `pyright` (`standard` mode), a dev dependency run via `uv run pyright`. Config lives in `[tool.pyright]` in `pyproject.toml`, targeting `src`, `tests`, and `tests-e2e`, pinned to the `.venv`.
- **Repo shape:**
  - `src/vision_modules/` — the package, one module per core concept.
  - `specs/` — pre-implementation design docs, one per concept (this folder).
  - `plans/` — implementation plans turning settled specs into buildable steps.
  - `tests/` at repo root, mirroring the `src/vision_modules/` module structure.
  - `tests-e2e/` at repo root, for the live tier above — not collected by the default `pytest` run.

## Open questions

None currently.
