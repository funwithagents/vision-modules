# Live-tier fixtures

`hand_<gesture>.jpg` — a real photo with one clearly visible hand making
`<gesture>`, where `<gesture>` is one of the HaGRID class names
(`specs/gesture_classifier.md`, e.g. `hand_palm.jpg`, `hand_stop.jpg`).

Used by:
- `tests-e2e/test_hand_live.py` and `tests-e2e/test_hand_pipeline_live.py` as
  the default hand-detection input (any one file; gesture doesn't matter
  there) when `VISION_MODULES_HAND_IMAGE` isn't set.
- `tests-e2e/test_hand_pipeline_live.py`'s classification test, parametrized
  over every `hand_<gesture>.jpg` found here, asserting the full pipeline
  actually predicts the photographed gesture — the filename is the ground
  truth label.

`no_hand.jpg` — a real photo with no hand in frame. Used by
`tests-e2e/test_hand_pipeline_live.py`'s negative-path test, asserting the
full pipeline reports `present=False` end to end instead of a stale or
fabricated result.

`two_hands.jpg` (optional) — a real photo with two clearly visible hands of
different apparent size. Used by `tests-e2e/test_hand_live.py` to check that the
detector returns two hands when asked for two and one when asked for one; the
test skips when the file is absent.

These are original photos contributed to the project (not sourced from a
third-party dataset), so they carry no external license/attribution
obligations — covered by this repo's own license like any other file here.
