# Multi-hand selection

**Status:** Done

Implements the settled behavior in `specs/hand.md` ("One count: `max_hands`", "Selection: rank, hysteresis, slot stability", "`HandStage`") and `specs/hand_demo.md` (per-slot classification panels, `--max-hands`). Delivers a `HandStage` with a single `max_hands` (default 2) that is both the detector's capacity and the publish cap, a pure selection rule `select_hands` that ranks by area with hysteresis and keeps slot order stable across frames, and a demo that shows one colour-linked classification panel per slot. Deliberately leaves out a detector capacity separate from `max_hands` (hand.md open question 2), a `track_id` surviving occlusion (hand.md open question 3), classifier batching (gesture_classifier.md open question 3), and any change to `GestureClassifier`, which already classifies every published hand.

## How to work this plan

- Read `AGENTS.md` first, then `specs/hand.md` and `specs/hand_demo.md`. The spec is the "what"; this plan is the "how". If the plan and the spec disagree, the spec wins — say so in your summary.
- Do the steps **in order**. Each step ends with a command to run; do not move on while it fails.
- Touch only the files listed in "Scope". Do not rename existing functions, classes or tests unless a step says so.
- The verification commands are always the same four, run from the repo root:

  ```
  uv run ruff check .
  uv run ruff format .
  uv run pyright
  uv run pytest
  ```

## Scope

- `src/vision_modules/hand.py` — add `MATCH_IOU`, `Rank`, `box_area`, `box_iou`, `select_hands`; change `HandStage.__init__` (new defaults and kwargs, validation, `_previous`), `HandStage.process` (use `select_hands`), `HandStage.close` (reset `_previous`).
- `src/vision_modules/__init__.py` — re-export `select_hands`.
- `tests/test_hand.py` — extend `ScriptedDetector`; add tests for `select_hands`, slot stability through the stage, reset across runs, validation; adjust one existing test.
- `examples/hand_demo.py` — `SLOT_COLORS`, `SLOT_NAMES`, `slot_color`, `slot_name`; `draw_boxes` per slot; `summarize(hr, gesture, index=0)`; `--max-hands`; one `gr.Label` per slot.
- `tests/test_hand_demo.py` — tests for `summarize` with an index and for `draw_boxes` colours.
- `tests-e2e/support.py` — `two_hands_image_path()`; `tests-e2e/test_hand_live.py` — one new test; `tests-e2e/fixtures/README.md` — describe the optional `two_hands.jpg`.
- `README.md` — Hand section and demo flags.
- `specs/hand.md`, `specs/hand_demo.md`, `specs/_index.md`, `plans/_index.md` — statuses at the very end (step 9).

## Steps

### Step 1 — pure helpers in `src/vision_modules/hand.py`

Add, right after the `HandDetector` protocol and before the `Hand` dataclass, the following. `Literal` and `Sequence` need importing (`from typing import Literal, Protocol` and `from collections.abc import Sequence`).

```python
MATCH_IOU = 0.3  # a candidate must overlap a previously published box this much to be its incumbent
Rank = Literal["area", "score"]


def box_area(box: tuple[float, float, float, float]) -> float:
    """Area of a normalized (x0, y0, x1, y1) box; 0.0 for an inverted box."""
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection over union of two normalized boxes; 0.0 when they don't overlap."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def select_hands(
    candidates: Sequence[DetectedHand],
    previous: Sequence[DetectedHand],
    max_hands: int,
    rank: Rank,
    hysteresis: float,
) -> tuple[DetectedHand, ...]:
    """Pick at most `max_hands` of `candidates`, slot-stable against `previous`.

    See specs/hand.md "Selection: rank, hysteresis, slot stability".
    """
    # 1. rank key per candidate
    keys = [box_area(d.box) if rank == "area" else d.score for d in candidates]

    # 2. match candidates to previously published slots by IoU, best pairs first
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(previous):
        for ci, c in enumerate(candidates):
            iou = box_iou(p.box, c.box)
            if iou >= MATCH_IOU:
                pairs.append((iou, pi, ci))
    pairs.sort(key=lambda t: t[0], reverse=True)
    slot_of: dict[int, int] = {}  # candidate index -> previous slot index
    used_slots: set[int] = set()
    for _, pi, ci in pairs:
        if pi in used_slots or ci in slot_of:
            continue
        slot_of[ci] = pi
        used_slots.add(pi)

    # 3. hysteresis: incumbents get a boost
    boosted = [
        k * (1.0 + hysteresis) if ci in slot_of else k for ci, k in enumerate(keys)
    ]

    # 4. choose: highest boosted key first; sorted() is stable, so ties keep detector order
    order = sorted(range(len(candidates)), key=lambda ci: boosted[ci], reverse=True)
    kept = order[:max_hands]

    # 5. order: incumbents in their previous slot order, then newcomers in rank order
    incumbents = sorted(
        (ci for ci in kept if ci in slot_of), key=lambda ci: slot_of[ci]
    )
    newcomers = [ci for ci in kept if ci not in slot_of]
    return tuple(candidates[ci] for ci in incumbents + newcomers)
```

Leave `padded_box` exactly as it is.

Run: `uv run ruff check . && uv run pyright` — both clean.

### Step 2 — `HandStage` in `src/vision_modules/hand.py`

Replace `HandStage.__init__`, `_detector`, `process` and `close` with:

```python
class HandStage(Stage[Frame, HandResult]):
    def __init__(
        self,
        provider: Upstream[Frame],
        target_fps: float | None = 30,
        pad: float = 0.35,
        max_hands: int = 2,
        rank: Rank = "area",
        hysteresis: float = 0.2,
        detector: HandDetector | None = None,
    ) -> None:
        if max_hands < 1:
            raise ValueError(f"max_hands must be >= 1, got {max_hands}")
        if hysteresis < 0:
            raise ValueError(f"hysteresis must be >= 0, got {hysteresis}")
        if rank not in ("area", "score"):
            raise ValueError(f"rank must be 'area' or 'score', got {rank!r}")
        super().__init__(provider, target_fps)
        self.pad = pad
        self.max_hands = max_hands
        self.rank: Rank = rank
        self.hysteresis = hysteresis
        self._borrowed = detector  # caller's: used, never closed here
        self._owned: HandDetector | None = (
            None  # ours: created per run, closed in close()
        )
        self._previous: tuple[
            DetectedHand, ...
        ] = ()  # last published, worker thread only

    def _detector(self) -> HandDetector:
        if self._borrowed is not None:
            return self._borrowed
        if self._owned is None:
            self._owned = MediaPipeHandDetector(num_hands=self.max_hands)
        return self._owned

    def process(self, item: Frame) -> HandResult:
        frame = item
        found = self._detector().detect(frame.image, frame.ts)
        selected = select_hands(
            found, self._previous, self.max_hands, self.rank, self.hysteresis
        )
        self._previous = selected
        height, width = frame.image.shape[:2]
        hands: list[Hand] = []
        for d in selected:
            x0, y0, x1, y1 = padded_box(d.box, width, height, self.pad)
            crop = frame.image[y0:y1, x0:x1].copy() if x1 > x0 and y1 > y0 else None
            hands.append(Hand((x0, y0, x1, y1), crop, d.score))
        return HandResult(
            frame.frame_id, frame.ts, present=bool(hands), hands=tuple(hands)
        )

    def close(self) -> None:
        self._previous = ()  # a restarted run starts with no incumbents
        if self._owned is not None:
            self._owned.close()
            self._owned = None  # a restarted run creates a fresh one
```

Notes:
- The validation runs **before** `super().__init__`, so a bad argument never creates a thread.
- `_previous` is only read and written inside `process()` and `close()`, which run on the worker thread (or after it has stopped), so it needs no lock.

Run: `uv run pytest tests/test_hand.py`. Expect **one failure**: `test_stage_orders_by_score_and_truncates_to_max_hands` (its three boxes are identical, so area ranking keeps detector order). Step 4 fixes it.

### Step 3 — export `select_hands`

In `src/vision_modules/__init__.py`, add `select_hands` to the `from vision_modules.hand import (...)` block (alphabetical: after `MediaPipeHandDetector`) and add `"select_hands"` to `__all__` right after `"select_device"`.

Run: `uv run ruff check . && uv run pyright` — clean.

### Step 4 — tests in `tests/test_hand.py`

**4a. Make `ScriptedDetector` play a sequence.** Replace the class with:

```python
class ScriptedDetector:
    """Returns a fixed tuple, or — given a list — one tuple per detect() call,
    repeating the last entry once the list is exhausted."""

    def __init__(
        self,
        hands: tuple[DetectedHand, ...] | list[tuple[DetectedHand, ...]],
    ) -> None:
        self.script = hands if isinstance(hands, list) else [hands]
        self.calls: list[float] = []
        self.closed = False

    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]:
        index = min(len(self.calls), len(self.script) - 1)
        self.calls.append(ts)
        return self.script[index]

    def close(self) -> None:
        self.closed = True
```

Every existing test passes a tuple, so nothing else changes.

**4b. Fix the one broken test.** In `test_stage_orders_by_score_and_truncates_to_max_hands`, construct the stage with `rank="score"`:

```python
stage = HandStage(
    provider, target_fps=None, max_hands=2, rank="score", detector=detector
)
```

**4c. Add shared boxes** near the top of the file (after `make_frame`). All boxes are normalized; areas are noted so the expected results are easy to check by hand:

```python
from vision_modules.hand import MATCH_IOU, box_area, box_iou, select_hands

A = DetectedHand((0.0, 0.0, 0.4, 0.4), 0.5)  # area 0.16
B = DetectedHand((0.6, 0.0, 0.9, 0.3), 0.9)  # area 0.09, top right
C = DetectedHand((0.0, 0.5, 0.5, 1.0), 0.1)  # area 0.25, bottom left
A_MOVED = DetectedHand(
    (0.02, 0.0, 0.42, 0.4), 0.5
)  # A shifted a little: area 0.16, IoU with A ≈ 0.9
B_GROWN = DetectedHand(
    (0.55, 0.0, 1.0, 0.45), 0.9
)  # B grown: area 0.2025, IoU with B ≈ 0.44
N_10 = DetectedHand(
    (0.55, 0.55, 0.97, 0.97), 0.5
)  # newcomer, area 0.1764 (10 % bigger than A)
N_30 = DetectedHand(
    (0.5, 0.5, 0.96, 0.96), 0.5
)  # newcomer, area 0.2116 (32 % bigger than A)
A_FAR = DetectedHand((0.5, 0.5, 0.9, 0.9), 0.5)  # same size as A, no overlap with A
```

**4d. Pure-function tests.** Add a section `# --- select_hands ---`:

```python
def test_box_area_and_iou() -> None:
    assert box_area(A.box) == pytest.approx(0.16)
    assert box_area((0.5, 0.5, 0.4, 0.4)) == 0.0  # inverted box
    assert box_iou(A.box, A.box) == pytest.approx(1.0)
    assert box_iou(A.box, C.box) == 0.0  # disjoint
    assert box_iou(B.box, B_GROWN.box) == pytest.approx(0.09 / 0.2025)
    assert (
        box_iou(B.box, B_GROWN.box) >= MATCH_IOU
    )  # the slot-stability tests rely on this


def test_select_no_previous_ranks_by_area_and_truncates() -> None:
    assert select_hands((B, A, C), (), 3, "area", 0.2) == (C, A, B)
    assert select_hands((B, A, C), (), 2, "area", 0.2) == (C, A)
    assert select_hands((B, A, C), (), 1, "area", 0.2) == (C,)


def test_select_rank_by_score() -> None:
    assert select_hands((B, A, C), (), 3, "score", 0.2) == (B, A, C)
    assert select_hands((A, C, B), (), 1, "score", 0.2) == (B,)


def test_select_ties_keep_detector_order() -> None:
    d1 = DetectedHand((0.0, 0.0, 0.3, 0.3), 0.5)
    d2 = DetectedHand((0.5, 0.5, 0.8, 0.8), 0.5)  # same area, same score
    assert select_hands((d1, d2), (), 2, "area", 0.2) == (d1, d2)
    assert select_hands((d2, d1), (), 2, "area", 0.2) == (d2, d1)


def test_select_hysteresis_keeps_the_incumbent_unless_clearly_beaten() -> None:
    previous = (A,)
    # 10 % bigger newcomer: kept out at 0.2, wins at 0
    assert select_hands((A_MOVED, N_10), previous, 1, "area", 0.2) == (A_MOVED,)
    assert select_hands((A_MOVED, N_10), previous, 1, "area", 0.0) == (N_10,)
    # 30 % bigger newcomer beats the 20 % boost
    assert select_hands((A_MOVED, N_30), previous, 1, "area", 0.2) == (N_30,)


def test_select_far_moved_hand_is_a_newcomer_not_an_incumbent() -> None:
    # A_FAR has no overlap with A, so it gets no boost and the bigger N_10 wins
    assert select_hands((A_FAR, N_10), (A,), 1, "area", 0.2) == (N_10,)


def test_select_slot_order_is_stable_when_sizes_cross() -> None:
    previous = (A, B)  # A in slot 0, B in slot 1
    # detector now returns them the other way round, and B has grown past A
    assert select_hands((B_GROWN, A_MOVED), previous, 2, "area", 0.2) == (
        A_MOVED,
        B_GROWN,
    )


def test_select_newcomer_fills_a_vacated_slot_after_the_incumbents() -> None:
    previous = (A_MOVED, B_GROWN)
    # B left; C appears, bigger than A. A keeps slot 0, C is appended.
    assert select_hands((C, A_MOVED), previous, 2, "area", 0.2) == (A_MOVED, C)
```

**4e. Stage-level tests.** Add to the `# --- HandStage ---` section. They use a 100×100 frame and `pad=0.0`, so a normalized box `(x0, y0, x1, y1)` becomes the pixel bbox `(100*x0, 100*y0, 100*x1, 100*y1)`.

```python
def test_stage_keeps_slot_order_across_frames() -> None:
    detector = ScriptedDetector([(A, B), (B_GROWN, A_MOVED)])
    provider = FakeProvider()
    provider.frame = make_frame(1, w=100, h=100)
    stage = HandStage(
        provider, target_fps=None, pad=0.0, max_hands=2, detector=detector
    )
    stage.start()
    try:
        wait_until(lambda: stage.published_count == 1)
        provider.frame = make_frame(2, w=100, h=100)
        wait_until(lambda: stage.published_count == 2)
        r = stage.latest()
        assert r is not None and r.frame_id == 2
        assert [h.bbox for h in r.hands] == [
            (2, 0, 42, 40),
            (55, 0, 100, 45),
        ]  # A first, then B
    finally:
        stage.stop()


def test_stage_forgets_incumbents_between_runs() -> None:
    # run 1 publishes A; run 2 sees A and a 10 % bigger newcomer with max_hands=1.
    # If the memory leaked across runs, A would be kept; after close() it must not be.
    detector = ScriptedDetector([(A,), (A_MOVED, N_10)])
    provider = FakeProvider()
    provider.frame = make_frame(1, w=100, h=100)
    stage = HandStage(
        provider, target_fps=None, pad=0.0, max_hands=1, detector=detector
    )
    stage.start()
    wait_until(lambda: stage.published_count == 1)
    stage.stop()

    provider.frame = make_frame(2, w=100, h=100)
    stage.start()
    try:
        wait_until(lambda: stage.published_count == 2)
        r = stage.latest()
        assert r is not None and r.first is not None
        assert r.first.bbox == (55, 55, 97, 97)  # N_10, not A_MOVED
    finally:
        stage.stop()


def test_stage_rejects_bad_arguments_before_starting() -> None:
    provider = FakeProvider()
    detector = ScriptedDetector(())
    with pytest.raises(ValueError):
        HandStage(provider, max_hands=0, detector=detector)
    with pytest.raises(ValueError):
        HandStage(provider, hysteresis=-0.1, detector=detector)
    with pytest.raises(ValueError):
        HandStage(provider, rank="size", detector=detector)  # pyright: ignore[reportArgumentType]
```

Leave `test_stage_owns_the_default_detector_one_per_run` untouched: it already asserts `created[0].num_hands == 2` for `max_hands=2`, which is exactly the "detector capacity == `max_hands`" rule.

Run: `uv run pytest tests/test_hand.py` — all pass. Then `uv run ruff check . && uv run ruff format . && uv run pyright`.

### Step 5 — demo code in `examples/hand_demo.py`

**5a. Slot colours.** Right after `DEMO_DIR = ...` add:

```python
# One colour per hand slot (BGR, for cv2). A slot's panel label names the same colour.
SLOT_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),  # green
    (255, 128, 0),  # blue
    (0, 165, 255),  # orange
    (255, 0, 255),  # magenta
)
SLOT_NAMES: tuple[str, ...] = ("green", "blue", "orange", "magenta")


def slot_color(index: int) -> tuple[int, int, int]:
    return SLOT_COLORS[index % len(SLOT_COLORS)]


def slot_name(index: int) -> str:
    return SLOT_NAMES[index % len(SLOT_NAMES)]
```

**5b. `draw_boxes`.** Replace with:

```python
def draw_boxes(canvas: np.ndarray, hr: HandResult | None) -> None:
    """One rectangle per published hand in its slot colour, with the slot number
    (1-based) tagged just inside the top-left corner so a box can be matched to its panel."""
    if hr is None:
        return
    for i, hand in enumerate(hr.hands):
        x0, y0, x1, y1 = hand.bbox
        color = slot_color(i)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            canvas,
            str(i + 1),
            (x0 + 4, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
```

**5c. `summarize`.** Replace with (note the new `index` parameter, default `0` so existing callers and tests keep working):

```python
def summarize(
    hr: HandResult | None, gesture: Gesture | None, index: int = 0
) -> dict[str, float]:
    """Per-class scores of hand slot `index` for a gr.Label (which sorts by score,
    so the top entry is always the top class); the validated one, if any, is
    marked with a checkmark.
    """
    if hr is None or index >= len(hr.hands):
        return {"no hand": 1.0}
    entry = (
        gesture.hands[index]
        if gesture is not None and index < len(gesture.hands)
        else None
    )
    if entry is None or not entry.scores:
        return {"...": 1.0}  # classifier hasn't caught up to the hand stage yet
    if entry.label is None:
        return dict(entry.scores)
    return {
        (f"✓ {label}" if label == entry.label else label): score
        for label, score in entry.scores.items()
    }
```

**5d. CLI.** In `parse_args` add, after `--threshold`:

```python
parser.add_argument("--max-hands", type=int, default=2)
```

and in `main()` build the stage with it:

```python
hands = HandStage(provider, target_fps=args.hand_fps, max_hands=args.max_hands)
```

**5e. `on_frame`.** Replace with (it must always return exactly `1 + args.max_hands` values):

```python
def on_frame(frame_rgb: np.ndarray) -> tuple[Any, ...]:
    source.push(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    frame = provider.latest()
    if frame is None:
        return (frame_rgb, *([{"no hand": 1.0}] * args.max_hands))
    canvas = frame.image.copy()  # NEVER draw on frame.image
    hr = hands.latest()
    draw_boxes(canvas, hr)
    gesture = gestures.latest()
    panels = [summarize(hr, gesture, i) for i in range(args.max_hands)]
    return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), *panels)
```

**5f. Layout.** Replace the third column

```python
with gr.Column():
    scores = gr.Label(label="Classifications")
```

with

```python
with gr.Column():
    score_panels = [
        gr.Label(label=f"Hand {i + 1} ({slot_name(i)})") for i in range(args.max_hands)
    ]
```

and change the stream binding's outputs from `outputs=[cam_out, scores]` to `outputs=[cam_out, *score_panels]`.

Run: `uv run ruff check . && uv run ruff format . && uv run pyright && uv run pytest tests/test_hand_demo.py` — clean (existing `summarize` tests still pass because `index` defaults to 0).

### Step 6 — demo tests in `tests/test_hand_demo.py`

Add `SLOT_COLORS`, `draw_boxes` to the `from hand_demo import (...)` list. Then add, after the existing `summarize` tests:

```python
def test_summarize_index_beyond_published_hands_is_no_hand() -> None:
    assert summarize(_hand_result(1), _gesture("palm", {"palm": 1.0}), index=1) == {
        "no hand": 1.0
    }


def test_summarize_index_with_hand_but_no_gesture_entry_is_pending() -> None:
    # two hands published, classifier result only has one entry so far
    assert summarize(_hand_result(2), _gesture("palm", {"palm": 1.0}), index=1) == {
        "...": 1.0
    }


def test_summarize_reads_the_requested_slot() -> None:
    two = Gesture(
        1,
        1.0,
        present=True,
        hands=(
            HandGesture("palm", 0.8, {"fist": 0.2, "palm": 0.8}),
            HandGesture("fist", 0.7, {"fist": 0.7, "palm": 0.3}),
        ),
    )
    assert summarize(_hand_result(2), two, index=0) == {"fist": 0.2, "✓ palm": 0.8}
    assert summarize(_hand_result(2), two, index=1) == {"✓ fist": 0.7, "palm": 0.3}


# --- draw_boxes ----------------------------------------------------------------


def test_draw_boxes_uses_one_colour_per_slot() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    hr = HandResult(
        1,
        1.0,
        present=True,
        hands=(Hand((10, 10, 50, 50), None, 0.9), Hand((60, 60, 90, 90), None, 0.9)),
    )
    draw_boxes(canvas, hr)
    assert (
        tuple(int(v) for v in canvas[10, 30]) == SLOT_COLORS[0]
    )  # on box 1's top edge
    assert (
        tuple(int(v) for v in canvas[60, 75]) == SLOT_COLORS[1]
    )  # on box 2's top edge
    assert not canvas[5, 5].any()  # outside both boxes: untouched


def test_draw_boxes_with_no_result_leaves_the_canvas_alone() -> None:
    canvas = np.zeros((20, 20, 3), np.uint8)
    draw_boxes(canvas, None)
    assert not canvas.any()
```

Run: `uv run pytest tests/test_hand_demo.py` — all pass.

### Step 7 — live tier (optional fixture, skips when absent)

**7a. `tests-e2e/support.py`.** Add:

```python
TWO_HANDS_IMAGE = FIXTURES_DIR / "two_hands.jpg"


def two_hands_image_path() -> Path:
    """Path to a real photo with two clearly visible hands. Skips if absent."""
    if TWO_HANDS_IMAGE.exists():
        return TWO_HANDS_IMAGE
    pytest.skip("no two-hand fixture available: add tests-e2e/fixtures/two_hands.jpg")
```

The name is deliberately **not** `hand_*.jpg`, so `labeled_hand_fixtures()` never treats it as a gesture ground truth.

**7b. `tests-e2e/test_hand_live.py`.** Add (import `two_hands_image_path` from `support` and `box_iou` from `vision_modules.hand`):

```python
def test_mediapipe_detector_returns_at_most_num_hands() -> None:
    img = cv2.imread(str(two_hands_image_path()))
    assert img is not None
    two = MediaPipeHandDetector(num_hands=2)
    one = MediaPipeHandDetector(num_hands=1)
    try:
        found_two = two.detect(img, 0.0)
        found_one = one.detect(img, 0.0)
    finally:
        two.close()
        one.close()
    assert len(found_two) == 2
    assert box_iou(found_two[0].box, found_two[1].box) < 0.5  # two distinct hands
    assert len(found_one) == 1  # never more than num_hands: the spec's one-count trade
```

**7c. `tests-e2e/fixtures/README.md`.** Add a paragraph:

```
`two_hands.jpg` (optional) — a real photo with two clearly visible hands of
different apparent size. Used by `tests-e2e/test_hand_live.py` to check that the
detector returns two hands when asked for two and one when asked for one; the
test skips when the file is absent.
```

Run: `uv run pytest tests-e2e/test_hand_live.py` — the new test **skips** (or passes if you added a photo); the two existing tests pass.

### Step 8 — `README.md`

In the "Hand" section replace the signature line

```
HandStage(provider, target_fps=30, pad=0.35, max_hands=1, detector: HandDetector | None = None)
```

with

```
HandStage(provider, target_fps=30, pad=0.35, max_hands=2, rank="area", hysteresis=0.2,
          detector: HandDetector | None = None)
```

Replace the bullet `- \`hands: tuple[Hand, ...]\` in detector order, highest score first` with:

```
- `hands: tuple[Hand, ...]`, at most `max_hands`. When a detector offers more candidates
  than that, the biggest (`rank="area"`) win, and a published hand is only displaced by one
  more than `hysteresis` bigger; the order is slot-stable, so `hands[i]` stays the same
  physical hand while it is visible. `max_hands` is also the detector's capacity: the shipped
  MediaPipe detector never returns more hands than that, so with `max_hands=1` it keeps the
  first hand it tracked.
```

In the demo section, wherever the CLI flags are listed, add `--max-hands` (default 2).

### Step 9 — statuses

Only after the full gate is clean:

- `specs/hand.md`: `**Status:** Updated` → `**Status:** Implemented`.
- `specs/hand_demo.md`: `**Status:** Updated` → `**Status:** Implemented`.
- `specs/_index.md`: both rows → `Implemented`.
- This file: `**Status:** Todo` → `**Status:** Done`; `plans/_index.md` row → `Done`.

## Verification

Definition of done — every box ticked:

- [ ] `uv run pytest tests/test_hand.py` passes, including the eight `select_hands` tests, `test_stage_keeps_slot_order_across_frames`, `test_stage_forgets_incumbents_between_runs` and the validation test.
- [ ] `uv run pytest tests/test_hand_demo.py` passes, including the three new `summarize` tests and the two `draw_boxes` tests.
- [ ] `uv run pytest tests-e2e/test_hand_live.py` runs with the new test skipped (no fixture) or passing (fixture present).
- [ ] Manual check: `uv run python examples/hand_demo.py`, hold up one hand, then bring in a second: two boxes in two colours, two panels, and the panels do not swap when you move the hands so their sizes cross. With `--max-hands 1`: one panel, and the first hand tracked keeps it until it leaves the frame (the documented MediaPipe behaviour).
- [ ] Full gate, all clean:

  ```
  uv run ruff check .
  uv run ruff format .
  uv run pyright
  uv run pytest
  ```

- [ ] Step 9 statuses flipped in both the files and the indexes.
