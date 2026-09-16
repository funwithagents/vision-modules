# Implementation plans

Implementation plans for Vision Modules — each plan turns a settled part of a spec (see [specs/_index.md](../specs/_index.md)) into concrete, buildable steps. Plans are ordered by their date-time filename prefix (`YYYYMMDDHHmm_`).

## Plans

<!-- One row per plan, chronological by filename prefix. Keep the Status column in sync with each plan's `**Status:**` line. -->

| Plan | Description | Status |
|---|---|---|
| [202609161500_dependency-extras.md](202609161500_dependency-extras.md) | Declare core runtime deps (`numpy`, `opencv-python`) and the feature-named `hand` extra in `pyproject.toml` | Done |
| [202609161510_pipeline-runtime.md](202609161510_pipeline-runtime.md) | `pipeline.py`: `LatestValue`, `Result`, `Stage`/`Module` worker loop, `Pipeline` lifecycle, with tests | Done |
| [202609161520_stream-provider.md](202609161520_stream-provider.md) | `stream.py`: `Frame`, `FrameSource`/`OpenCVSource`, `StreamProvider` capture thread, fast tests + camera live test | Done |

## Status legend

- **Todo** — written, not yet started
- **In progress** — actively being implemented
- **Done** — implemented, verified (lint/type-check/tests pass), and merged
