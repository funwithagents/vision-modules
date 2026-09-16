# Gesture classifier

**Status:** Done

Implements `specs/gesture_classifier.md` in full: `HandGesture` / `Gesture`, the `ImageClassifier` seam, `select_device`, the `GestureClassifier` module, and the shipped `HaGRIDViTClassifier` on Hugging Face `transformers`. Delivers `src/vision_modules/gesture_classifier.py`, fast tests with a scripted classifier, and an opt-in live test. The demo that shows it on a webcam is the **next** plan.

## Context you need before starting

- Read `AGENTS.md`, `specs/gesture_classifier.md` (all of it), `specs/testing.md`.
- Prerequisites: plans `…1500` through `…1530` are `Done`. Uses `Module`, `Result` from `vision_modules.pipeline` and `HandResult`, `Hand` from `vision_modules.hand`.
- `src/vision_modules/gesture_classifier.py` is a docstring-only stub; replace its body.
- **`torch`, `transformers`, `PIL` are optional-extra packages.** Import them **inside** `select_device` / `HaGRIDViTClassifier`, never at module top level. The fast tests never construct `HaGRIDViTClassifier`.
- Installed versions are `transformers 5.17`, `torch 2.14` (verified 2026-09-16).
- **`top_k` gotcha (verified in the installed source):** the image-classification pipeline *ignores* `top_k=None` and falls back to its default of 5 results. To get every class you must pass `top_k=<number of labels>` (the pipeline clamps anything larger to that number). The code below does this; do not "simplify" it to `top_k=None`.

## Scope

- `src/vision_modules/gesture_classifier.py` — everything below.
- `tests/test_gesture_classifier.py` — module logic with a scripted classifier.
- `tests-e2e/test_gesture_classifier_live.py` — real model, gated by `VISION_MODULES_E2E`.
- `specs/gesture_classifier.md` — frontmatter `tests:` gets both paths; status `Stable` → `Implemented` at the end (file + `specs/_index.md`).
- `plans/_index.md` + this file — status flips.

## Target API (exact names)

```python
@dataclass(frozen=True)
class HandGesture:
    label: (
        str | None
    )  # top class if its score >= threshold, else None; None when no crop
    confidence: float  # top score, 0.0 when no crop
    scores: dict[str, float]  # every class -> score; {} when no crop


@dataclass(frozen=True, eq=False)
class Gesture(Result):
    hands: tuple[HandGesture, ...]  # index-aligned with HandResult.hands

    @property
    def first(self) -> HandGesture | None: ...


class ImageClassifier(Protocol):
    @property
    def labels(self) -> tuple[str, ...]: ...
    def classify(self, image_bgr: np.ndarray) -> dict[str, float]: ...
    def close(self) -> None: ...


def select_device(preferred: str | None = None) -> str:
    """preferred if given; else 'mps' if torch.backends.mps.is_available(), else 'cuda' if torch.cuda.is_available(), else 'cpu'."""


class GestureClassifier(Module[HandResult, Gesture]):
    def __init__(
        self,
        hand_stage: Upstream[HandResult],
        target_fps: float | None = 5,
        threshold: float = 0.55,
        classifier: ImageClassifier | None = None,
        device: str | None = None,
    ) -> None: ...
    def process(self, hands: HandResult) -> Gesture: ...
    def close(self) -> None: ...  # closes the classifier if one was created


class HaGRIDViTClassifier:
    MODEL_ID = "dima806/hand_gestures_image_detection"

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None: ...
    @property
    def labels(self) -> tuple[str, ...]: ...
    def classify(self, image_bgr: np.ndarray) -> dict[str, float]: ...
    def close(
        self,
    ) -> None: ...  # drop the pipeline reference (no explicit close in transformers)
```

### `GestureClassifier.process`

1. Lazy: `if self._clf is None: self._clf = self._given or HaGRIDViTClassifier(device=select_device(self._device))`.
2. For each `hand` in `hands.hands`: if `hand.crop is None` → `HandGesture(None, 0.0, {})`; else `scores = self._clf.classify(hand.crop)`; `label, conf = max(scores.items(), key=lambda kv: kv[1])`; `HandGesture(label if conf >= threshold else None, conf, scores)`.
3. Return `Gesture(hands.frame_id, hands.ts, present=bool(out), hands=tuple(out))`.

Note `>=`: a score exactly equal to the threshold **clears** it (test 3 pins this).

### `HaGRIDViTClassifier`

```python
from transformers import pipeline as hf_pipeline
from PIL import Image

self._pipe = hf_pipeline("image-classification", model=model_id, device=device)
id2label = self._pipe.model.config.id2label  # {0: "call", 1: "dislike", ...}
self._labels = tuple(id2label[i] for i in sorted(id2label))
```

`classify()`: `pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))`; `preds = self._pipe(pil, top_k=len(self._labels))` → list of `{"label": str, "score": float}`, one per class; return `{p["label"]: float(p["score"]) for p in preds}`. The live test asserts the dict has exactly one entry per label — if it has 5, you passed the wrong `top_k`.

## Steps

1. Mark this plan `In progress` (file + index).
2. Sanity-check the installed pipeline (no download): `uv run python -c "from transformers.pipelines import ImageClassificationPipeline as P; import inspect; print(inspect.signature(P.postprocess))"` — expected to print a signature containing `top_k=5`, confirming the gotcha above. Nothing to adapt.
3. Write `src/vision_modules/gesture_classifier.py` per the API above.
4. Create `tests/test_gesture_classifier.py` (below); add its path under `tests:` in `specs/gesture_classifier.md`.
5. Create `tests-e2e/test_gesture_classifier_live.py` (below); add its path too.
6. Verification gate; then `VISION_MODULES_E2E=1 uv run pytest tests-e2e -q -k gesture` once (downloads ~340 MB, needs network) — must pass; without the variable it must skip.
7. Flip `specs/gesture_classifier.md` to `Implemented` (file + index); this plan to `Done` (file + index).

## Tests to write (`tests/test_gesture_classifier.py`)

Helpers: `class ScriptedClassifier` with `labels = ("fist", "palm", "stop")`, a `scores` dict it returns from `classify()`, a `seen: list[np.ndarray]` of crops it was given, a `closed` flag; `class FakeHandStage` with settable `.result`; `make_hand_result(frame_id, crops: list[np.ndarray | None])` building a `HandResult` with `Hand((0,0,8,8), crop, 0.9)` per crop; `wait_until` (copy).

1. `test_labels_the_top_class_above_threshold` — scores `{"fist": 0.1, "palm": 0.8, "stop": 0.1}`, threshold 0.55 → `first.label == "palm"`, `confidence == 0.8`, `scores` equals the dict.
2. `test_returns_none_label_below_threshold_but_keeps_scores` — `{"fist": 0.4, "palm": 0.35, "stop": 0.25}` → `label is None`, `confidence == 0.4`, `scores` intact.
3. `test_score_equal_to_threshold_clears_it` — top score `0.55`, threshold `0.55` → label is the top class.
4. `test_hand_without_crop_gets_empty_gesture` — crops `[None]` → `HandGesture(None, 0.0, {})`, and `classifier.seen == []`.
5. `test_empty_hand_result_gives_present_false` — no hands → `present is False`, `hands == ()`, `first is None`, same `frame_id`/`ts` as the input.
6. `test_multi_hand_output_is_index_aligned` — crops `[a, b]` with a classifier whose result depends on the crop (e.g. it returns `{"palm": 1.0}` if `crop[0,0,0] == 1` else `{"fist": 1.0}`); `hands[0].label == "palm"`, `hands[1].label == "fist"`.
7. `test_classifier_receives_the_crop_unchanged` — `np.array_equal(classifier.seen[0], crop)`.
8. `test_closes_classifier_on_stop` — `classifier.closed is True` after `stop()`.
9. `test_select_device_honours_preference` — `select_device("cpu") == "cpu"`; `select_device() in {"mps", "cuda", "cpu"}`. (This imports `torch`, which is slow on first import; acceptable — it is installed by the `dev` group.)

## Live test (`tests-e2e/test_gesture_classifier_live.py`)

```python
def test_hagrid_model_loads_and_scores_every_class():
    require_env("VISION_MODULES_E2E")
    clf = HaGRIDViTClassifier(
        device="cpu"
    )  # cpu: deterministic and available everywhere
    try:
        assert len(clf.labels) == 18 and {"palm", "stop", "fist"} <= set(clf.labels)
        scores = clf.classify(np.full((224, 224, 3), 127, np.uint8))
        assert set(scores) == set(clf.labels)
        assert abs(sum(scores.values()) - 1.0) < 1e-3
    finally:
        clf.close()
```

## Verification

```
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv run pytest tests-e2e -q                              # gesture live test skipped
VISION_MODULES_E2E=1 uv run pytest tests-e2e -q -k gesture
```

Observed 18 labels (matches `specs/gesture_classifier.md`):

`call`, `dislike`, `fist`, `four`, `like`, `mute`, `ok`, `one`, `palm`, `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`, `two_up_inverted`
