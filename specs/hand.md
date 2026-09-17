---
code:
  - src/vision_modules/hand.py
tests:
  - tests/test_hand.py
  - tests-e2e/test_hand_live.py
  - tests-e2e/test_hand_pipeline_live.py
---

# Hand

**Status:** Implemented

## Purpose

The shared hand front-end: one stage that finds the hands in each frame and cuts a padded crop per hand, **once per frame**, and publishes them as a `HandResult` for every hand-based module to consume. It exists so a module that needs a hand image ([gesture_classifier.md](gesture_classifier.md) today, any future hand module) gets an accurate, already-cropped input without running its own detection. Its job is **detect, select and crop** — nothing more is extracted from the detector in this version. It is a `Stage` from [pipeline.md](pipeline.md) whose upstream is the [stream.md](stream.md) provider.

## Decided

### One count: `max_hands`

`max_hands` (default **2**) is both how many hands the owned detector is asked to find (`MediaPipeHandDetector(num_hands=max_hands)`) and the cap on how many hands are published in `HandResult.hands` — and therefore how many hands downstream modules ([gesture_classifier.md](gesture_classifier.md)) classify. One knob, deliberately: a caller who wants two hands classified sets 2 and gets two detected, two published, two classified.

The trade this makes with the shipped backend must be understood: MediaPipe **tracks** in `VIDEO` mode and only re-runs palm detection while it is tracking fewer than `num_hands` hands, and it never returns more hands than `num_hands`. So with `max_hands=1` the stage sees exactly one candidate — the first hand MediaPipe locked onto — and a second, bigger hand entering the frame is not considered until the first is lost; the selection rule below cannot help, because it has nothing to choose among. With `max_hands=2` (the default) both hands of one person are detected and published, which is the case that matters. Choosing the best `N` out of more than `N` tracked hands would need a separate detector capacity (open question 2); it is not built.

### `Hand` and `HandResult`

`Hand` — frozen dataclass, one detected hand:

| Field | Type | Meaning |
|---|---|---|
| `bbox` | `tuple[int, int, int, int]` | `(x0, y0, x1, y1)` in full-frame pixels, **padded and clamped** to the frame |
| `crop` | `np.ndarray \| None` | BGR `image[y0:y1, x0:x1]` of the padded box, **an independent copy**; `None` only if the clamped box has zero area (hand at the frame edge) |
| `score` | `float` | the detector's per-hand score — for `MediaPipeHandDetector` this is the **handedness** (left/right) confidence, the only per-hand score the Tasks API exposes, so it is a weak signal of "how sure is the detector this is a hand" and is **not** the default ranking key |

`HandResult(Result)` — frozen: `frame_id`, `ts`, `present`, plus:

- `hands: tuple[Hand, ...]` — at most `max_hands` hands, chosen and ordered by the selection rule below. Empty when none; `present == bool(hands)`.
- `first: Hand | None` — convenience: `hands[0]` or `None`.

- **The crop is copied** before publishing: a slice is a view into the shared frame buffer ([stream.md](stream.md)); copying is cheap (a small region) and lets a classifier preprocess freely.
- **Nothing else from the detector is published.** Landmarks and handedness are computed by MediaPipe on the way to the box but stay internal. If a later module needs them, they are added to `Hand` as new optional fields — an additive, non-breaking change (open question 1).

### Selection: rank, hysteresis, slot stability

Whatever the detector returns, the stage decides which candidates are published and in what order with one pure rule, `select_hands(candidates, previous, max_hands, rank, hysteresis)`, applied every frame. It matters in two situations: a detector that returns **more** candidates than `max_hands` (a borrowed detector, or a future backend with a real detection score — never the owned MediaPipe one, see above), where it picks which ones; and every multi-hand frame, where it fixes the **order** so `hands[i]` does not jump between physical hands.

1. **Rank key.** Each candidate gets a key by `rank`:
   - `"area"` (default) — the area of its **unpadded normalized** box, `(x1 - x0) * (y1 - y0)`. The biggest hand is the closest hand, which is almost always the one the user means. Normalized area is aspect-agnostic; all candidates on a frame share the same frame, so it ranks the same as pixel area.
   - `"score"` — `DetectedHand.score`, for a detector whose score is a real detection confidence.
2. **Match against the previous frame.** Candidates are matched to the hands the stage **published on its previous processed frame** (kept as their unpadded normalized boxes) by intersection-over-union, greedily from the highest IoU pair down, each side used at most once, and only for pairs with IoU `>= MATCH_IOU` (`0.3`, a module constant, not a knob). A matched candidate is an **incumbent** of the slot its previous box occupied; every other candidate is a **newcomer**.
3. **Hysteresis.** An incumbent's key is multiplied by `(1 + hysteresis)` before comparing. With the default `hysteresis=0.2`, a newcomer must be more than 20 % bigger (or higher-scoring) than a published hand to displace it, so two similar hands do not flicker in and out of the result from frame to frame. `hysteresis=0` gives pure ranking. Must be `>= 0`.
4. **Choose.** Sort all candidates by their (boosted) key, descending, stable on detector order for ties; keep the first `max_hands`.
5. **Order: slot-stable.** The kept **incumbents keep their previous slot order** (relative order among themselves); the kept **newcomers follow, in rank order**. So `hands[i]` stays the same physical hand as long as it stays detected and selected — a two-hand consumer (one panel per hand in [hand_demo.md](hand_demo.md), a per-hand file from `save_crops` in [snapshot.md](snapshot.md)) does not see the two hands trade indices because their sizes crossed. The rank order is therefore what a set of hands is published in when it **first** appears; after that, order is stability, not rank.

Consequences worth naming:

- With `max_hands=1` and a detector that returns several candidates, this is exactly "pick the biggest hand, and keep it until a clearly bigger one shows up". With the owned MediaPipe detector at `max_hands=1` there is only ever one candidate, so the rule is a pass-through.
- Slot stability is **not** identity across a gap: when a hand leaves and its slot is vacated, the next newcomer may take that index. A stable `track_id` that survives occlusion is still open question 3. There is no time decay: "previous frame" is the last frame this stage processed, however long ago; MediaPipe's own tracking has re-detected by then anyway.
- The memory of the previous frame is per run: cleared in `close()`, and naturally empty after a `present=False` frame (nothing was published to match against).
- The stage applies this rule to whatever the detector returns, borrowed or owned, so it is testable with a scripted detector and is not tied to MediaPipe.

### `HandStage`

```
HandStage(provider, target_fps: float | None = 30, pad: float = 0.35, max_hands: int = 2,
          rank: Literal["area", "score"] = "area", hysteresis: float = 0.2,
          detector: HandDetector | None = None)
```

- `Stage[Frame, HandResult]`. Per new frame: run the detector once with the frame's `ts`; select up to `max_hands` candidates with the rule above; for every selected hand, scale its normalized box to pixels, pad it by `pad` × box size on every side, clamp to the frame, copy the crop (or `None` for a degenerate box), build a `Hand`; publish a `HandResult` — also when no hand was found (`present=False`), so modules know the frame was seen and empty.
- **Validation** at construction: `max_hands >= 1`, `hysteresis >= 0`, `rank` one of the two literals — `ValueError` otherwise.
- **One detector, touched on the worker thread only.** With no `detector` given, the stage creates a `MediaPipeHandDetector(num_hands=max_hands)` lazily on its worker thread (MediaPipe objects are not thread-safe; no other thread may touch it), closes it in `close()` and creates a fresh one on the next run. A `detector` passed in is **borrowed** ([pipeline.md](pipeline.md) "Owned vs. borrowed backends"): the stage never closes it, so it survives `stop()`/`start()` cycles and the caller closes it when done; its capacity is the caller's business, and the stage still publishes at most `max_hands` of what it returns.
- **Cache directory:** the downloaded model bundle lives under `~/.cache/vision-modules/`, overridable with the `VISION_MODULES_CACHE` environment variable (no `platformdirs` dependency); `model_path=` bypasses the cache entirely. Shared convention with [gesture_classifier.md](gesture_classifier.md).

### `HandDetector` protocol — the seam

```
class HandDetector(Protocol):
    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]: ...
    def close(self) -> None: ...

DetectedHand: box (x0, y0, x1, y1) normalized to [0, 1], unpadded; score float
```

- The seam is deliberately **boxes only**. Any hand detector (MediaPipe today, a YOLO-style detector tomorrow) can sit behind it, because the stage needs nothing but a box and a score.
- It lets the stage's geometry (scaling, padding, clamping, copy, degenerate box, `present=False` path) **and its selection rule** (rank, hysteresis, slot stability, `max_hands`) be tested in `tests/` with a scripted detector returning known boxes per call — no model, no network, per [testing.md](testing.md). The real detector runs only in `tests-e2e/`.

### `MediaPipeHandDetector` — the shipped detector (Tasks API)

- Built on the **MediaPipe Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`), not the legacy `mp.solutions.hands` (frozen, no longer maintained). MediaPipe exposes no standalone hand-box detector, so the landmarker is run and the box is taken as the extremes of its 21 landmarks — internally; landmarks never leave the adapter.
- **Running mode `VIDEO`**: the detector is fed the frame's `ts` (converted to strictly increasing integer milliseconds) so MediaPipe can track between frames instead of re-detecting from scratch every time. `IMAGE` mode would be stateless and slower; `LIVE_STREAM` mode's async callback is redundant with our own threading. Tracking is also why the detector never offers the stage a choice (see "One count" above): palm detection re-runs only while fewer than `num_hands` hands are tracked, and at most `num_hands` are ever returned.
- Options exposed as kwargs, defaulting to MediaPipe's: `num_hands` (from the stage's `max_hands`), `min_hand_detection_confidence`, `min_hand_presence_confidence`, `min_tracking_confidence`. The reference scripts ran the legacy API at 0.6; tune on real use.
- Converts BGR→RGB and wraps in `mp.Image` internally; maps each native hand to a `DetectedHand` (landmark extremes → normalized box, handedness score → `score`). The Tasks result carries no per-hand detection confidence, hence `score`'s caveat above.
- **Model bundle** (`hand_landmarker.task`, a few MB): **not bundled in the wheel**; downloaded from Google's model storage on first use into a user cache directory, overridable with `model_path=`. Same lazy-download pattern as the Hugging Face model in [gesture_classifier.md](gesture_classifier.md). The exact URL is pinned in code, at plan time.
- Installed through the `hand` extra ([project.md](project.md)).

## Open questions

1. **Exposing landmarks / handedness later.** A future geometry module (finger counting, pointing direction) would need MediaPipe's 21 landmarks and `"Left"` / `"Right"` label. Plan: add them as optional fields on `Hand` and widen `DetectedHand` when that module is specced; note MediaPipe labels handedness assuming a mirrored image, so [stream.md](stream.md)'s `mirror` option becomes relevant then. Deferred — out of v1.
2. **Detector capacity separate from `max_hands`.** Asking MediaPipe for more hands than are published (`num_hands = max(2, max_hands)`, say) would let the stage pick the biggest one at `max_hands=1` instead of the first one tracked, at the cost of a second knob and one more landmark pass per frame. Rejected for now to keep a single count; revisit if a single-hand consumer needs "the closest hand" rather than "the first hand seen".
3. **Hand identity across gaps.** Slot stability (above) keeps `hands[i]` the same hand while it stays detected, but a slot is reused once vacated, and nothing survives an occlusion. A stable `track_id` on `Hand` (with a short grace period before a slot is released) would close this. Deferrable — matters only for a consumer that must follow a specific hand through a dropout.
4. **Tuning `MATCH_IOU`.** `0.3` assumes a hand moves less than most of its own width between two processed frames, which holds at the default 30 fps but may not at a heavily down-sampled hand stage (a few fps). If mismatches show up in practice, either lower it, switch the match to center distance relative to box size, or promote it to a constructor knob. Deferrable until observed.
