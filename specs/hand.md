---
code:
  - src/vision_modules/hand.py
tests:
  - tests/test_hand.py
  - tests-e2e/test_hand_live.py
---

# Hand

**Status:** Implemented

## Purpose

The shared hand front-end: one stage that finds the hands in each frame and cuts a padded crop per hand, **once per frame**, and publishes them as a `HandResult` for every hand-based module to consume. It exists so a module that needs a hand image ([gesture_classifier.md](gesture_classifier.md) today, any future hand module) gets an accurate, already-cropped input without running its own detection. Its job is **detect and crop** — nothing more is extracted from the detector in this version. It is a `Stage` from [pipeline.md](pipeline.md) whose upstream is the [stream.md](stream.md) provider.

## Decided

### `Hand` and `HandResult`

`Hand` — frozen dataclass, one detected hand:

| Field | Type | Meaning |
|---|---|---|
| `bbox` | `tuple[int, int, int, int]` | `(x0, y0, x1, y1)` in full-frame pixels, **padded and clamped** to the frame |
| `crop` | `np.ndarray \| None` | BGR `image[y0:y1, x0:x1]` of the padded box, **an independent copy**; `None` only if the clamped box has zero area (hand at the frame edge) |
| `score` | `float` | detector confidence for this hand |

`HandResult(Result)` — frozen: `frame_id`, `ts`, `present`, plus:

- `hands: tuple[Hand, ...]` — every hand found, in the detector's order (highest score first). Empty when none; `present == bool(hands)`.
- `first: Hand | None` — convenience: `hands[0]` or `None`.

Multi-hand is therefore a shape decision made once: today `max_hands=1` and modules read `first`; raising `max_hands` later changes nothing in the contract.

- **The crop is copied** before publishing: a slice is a view into the shared frame buffer ([stream.md](stream.md)); copying is cheap (a small region) and lets a classifier preprocess freely.
- **Nothing else from the detector is published.** Landmarks and handedness are computed by MediaPipe on the way to the box but stay internal. If a later module needs them, they are added to `Hand` as new optional fields — an additive, non-breaking change (open question 1).
- **No identity across frames.** `hands[i]` on one frame is not guaranteed to be the same physical hand as `hands[i]` on the next (open question 2).

### `HandStage`

```
HandStage(provider, target_fps: float | None = 30, pad: float = 0.35, max_hands: int = 1,
          detector: HandDetector | None = None)
```

- `Stage[Frame, HandResult]`. Per new frame: run the detector once with the frame's `ts`; for every hand found, scale its normalized box to pixels, pad it by `pad` × box size on every side, clamp to the frame, copy the crop (or `None` for a degenerate box), build a `Hand`; publish a `HandResult` — also when no hand was found (`present=False`), so modules know the frame was seen and empty.
- **Owns the only detector instance**, created and used on its worker thread only. MediaPipe objects are not thread-safe; no other thread may touch it.

### `HandDetector` protocol — the seam

```
class HandDetector(Protocol):
    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]: ...
    def close(self) -> None: ...

DetectedHand: box (x0, y0, x1, y1) normalized to [0, 1], unpadded; score float
```

- The seam is deliberately **boxes only**. Any hand detector (MediaPipe today, a YOLO-style detector tomorrow) can sit behind it, because the stage needs nothing but a box and a score.
- It lets the stage's geometry (scaling, padding, clamping, copy, degenerate box, `present=False` path, `max_hands`) be tested in `tests/` with a scripted detector returning known boxes — no model, no network, per [testing.md](testing.md). The real detector runs only in `tests-e2e/`.

### `MediaPipeHandDetector` — the shipped detector (Tasks API)

- Built on the **MediaPipe Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`), not the legacy `mp.solutions.hands` (frozen, no longer maintained). MediaPipe exposes no standalone hand-box detector, so the landmarker is run and the box is taken as the extremes of its 21 landmarks — internally; landmarks never leave the adapter.
- **Running mode `VIDEO`**: the detector is fed the frame's `ts` (converted to strictly increasing integer milliseconds) so MediaPipe can track between frames instead of re-detecting from scratch every time. `IMAGE` mode would be stateless and slower; `LIVE_STREAM` mode's async callback is redundant with our own threading.
- Options exposed as kwargs, defaulting to MediaPipe's: `num_hands` (from `max_hands`), `min_hand_detection_confidence`, `min_hand_presence_confidence`, `min_tracking_confidence`. The reference scripts ran the legacy API at 0.6; tune on real use.
- Converts BGR→RGB and wraps in `mp.Image` internally; maps each native hand to a `DetectedHand` (landmark extremes → normalized box, handedness score → `score`).
- **Model bundle** (`hand_landmarker.task`, a few MB): **not bundled in the wheel**; downloaded from Google's model storage on first use into a user cache directory, overridable with `model_path=`. Same lazy-download pattern as the Hugging Face model in [gesture_classifier.md](gesture_classifier.md). The exact URL is pinned in code, at plan time.
- Installed through the `hand` extra ([project.md](project.md)).

## Open questions

1. **Exposing landmarks / handedness later.** A future geometry module (finger counting, pointing direction) would need MediaPipe's 21 landmarks and `"Left"` / `"Right"` label. Plan: add them as optional fields on `Hand` and widen `DetectedHand` when that module is specced; note MediaPipe labels handedness assuming a mirrored image, so [stream.md](stream.md)'s `mirror` option becomes relevant then. Deferred — out of v1.
2. **Hand identity across frames.** Whether to give each `Hand` a stable `track_id` (nearest-bbox matching between frames) so a two-hand consumer can follow a hand over time. Deferrable — matters only once `max_hands > 1` is used in practice.
3. **Cache directory convention.** Decided: `~/.cache/vision-modules/`, overridable with the `VISION_MODULES_CACHE` environment variable; no `platformdirs` dependency. Shared with [gesture_classifier.md](gesture_classifier.md).
4. **Python 3.12 wheel availability** of `mediapipe` on macOS arm64 must be confirmed at plan time (it is a hard dependency of the `hand` extra).
