---
code:
  - src/vision_modules/gesture_classifier.py
tests:
  - tests/test_gesture_classifier.py
  - tests-e2e/test_gesture_classifier_live.py
  - tests-e2e/test_hand_pipeline_live.py
---

# Gesture classifier

**Status:** Implemented

## Purpose

A perception module that names the gesture each hand is making, by classifying the hand crop with an image model trained on the **HaGRID** gesture vocabulary. It is the first (and, in this version, only) perception module: the shared hand stage ([hand.md](hand.md)) finds and crops the hand, this module says which HaGRID class the crop shows. It reports labels and scores only; what a label triggers is the consumer's call ([pipeline.md](pipeline.md), "the library perceives; the caller decides"). A `Module` from [pipeline.md](pipeline.md) consuming `Hand.crop` — which is exactly why the shared stage exists: the model is a classifier, not a detector, and needs a hand-cropped image to be accurate.

## Decided

### Vocabulary: HaGRID's classes, verbatim

Labels are the model's raw HaGRID class names, never renamed or remapped by the library. The 18 HaGRID classes are:

`call`, `dislike`, `fist`, `four`, `like`, `mute`, `ok`, `one`, `palm`, `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`, `two_up_inverted`

The runtime authority is the classifier's `labels` property (read from the model config), so a fine-tuned or different backend with another vocabulary works unchanged. A consumer that wants "number 2" or "open hand" maps the class names itself.

### Output

`HandGesture` — frozen dataclass, one hand:

- `label: str | None` — top class when its score clears `threshold`, else `None` ("no confident gesture"). Also `None` when the hand has no crop.
- `confidence: float` — score of the top class (`0.0` when no crop).
- `scores: dict[str, float]` — full label → score map from the model (empty when no crop), so a consumer can apply its own rule.

`Gesture(Result)` — frozen: `frame_id`, `ts`, `present`, plus:

- `hands: tuple[HandGesture, ...]` — index-aligned with `HandResult.hands`, so it inherits [hand.md](hand.md)'s slot stability: `hands[i]` follows the same physical hand while that hand stays published. Multi-hand classification needs nothing from this module beyond what it already does — one crop, one `HandGesture`, per published hand, up to the hand stage's `max_hands`.
- `first: HandGesture | None` — convenience: `hands[0]` or `None`.

### Module

```
GestureClassifier(hand_stage, target_fps: float | None = 5, threshold: float = 0.55,
                  classifier: ImageClassifier | None = None, device: str | None = None)
```

- `Module[HandResult, Gesture]`. `process` classifies every hand's crop (one call per crop, no batching); an empty `HandResult` gives an empty, `present=False` result.
- **Model loaded once per run**, lazily on the worker thread (a resource owned by that thread, per [pipeline.md](pipeline.md)); `close()` releases it and the next run loads a fresh one. A `classifier` passed in is **borrowed** ([pipeline.md](pipeline.md) "Owned vs. borrowed backends"): never closed by the module, reused across restarts, closed by the caller.
- **Device selection:** `"mps"` if available, else `"cuda"` if available, else `"cpu"` — a `select_device()` helper, overridable through `device`. The models here are small enough that CPU is a valid fallback and sidesteps MPS gaps (`PYTORCH_ENABLE_MPS_FALLBACK=1` is the user's escape hatch, never set by the library).

### `ImageClassifier` protocol — the seam

```
class ImageClassifier(Protocol):
    labels: tuple[str, ...]                                              # the vocabulary
    def classify(self, image_bgr: np.ndarray) -> dict[str, float]: ...   # label -> score, all classes
    def close(self) -> None: ...
```

- **`HaGRIDViTClassifier`** is the shipped implementation: Hugging Face `dima806/hand_gestures_image_detection`, a ViT fine-tuned on HaGRID (18 classes; ~85.8M params; Apache-2.0; ~96% reported accuracy), loaded through the `transformers` image-classification pipeline and queried for every class (`top_k` set to the number of labels — the pipeline ignores `top_k=None` and would return only its default top 5). First run downloads ~340 MB (cached by HF afterwards). Converts BGR→RGB (PIL) before inference. Installed through the `hand` extra ([project.md](project.md)).
- The protocol lets `tests/` drive the module with a scripted classifier (fixed score maps) — no weights, no network, per [testing.md](testing.md). The real model runs only in `tests-e2e/`. Those tests need no credential, only network on the first run to download the weights; an offline first run **fails** rather than skips (the model tier is "network, not keys" — see [testing.md](testing.md)).

### Not in scope: temporal / video models

The model takes single cropped images, so none of the `transformers` video utilities (`load_video`, `AutoVideoProcessor`, frame sampling by `fps` / `num_frames`) are needed. They would matter only for a clip-level temporal recognizer added later, and even then they are clip frame-samplers, not a live-stream orchestrator — the [pipeline.md](pipeline.md) graph stays our own code.

## Open questions

1. **Confidence threshold.** Starting value 0.55 (from the reference script); tune on real use. Deferrable (constructor kwarg).
2. **Framerate.** Starting value 5 fps against a 30 fps hand stage. Deferrable (constructor kwarg).
3. **Batching across hands.** With `max_hands > 1` (now the default in [hand.md](hand.md)), classifying the crops in one batched call would be cheaper than one call each. Deferrable: at 5 fps and two hands the per-call overhead is not what bounds the demo; revisit if a consumer raises `max_hands` and the classifier's achieved fps falls short of its target.
4. **Temporal smoothing.** A per-frame label can flicker between neighbouring classes; a consumer-side or later stabilizer concept could debounce it. Deferrable.
