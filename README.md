# Vision Modules

Composable vision modules over a live video stream, giving robots, systems, and agents
perception they can act on.

Perception here is modular: every module solves one perceptual job over the same incoming
stream and exposes it behind a uniform interface (`start()` / `stop()` / `latest()`). Richer
perception means switching on another module, not rewriting the pipeline. Shared work (finding
the hand, for instance) is done once in a shared stage and fanned out to every module that
needs it, and each node runs on its own thread at its own framerate.

The library **perceives; the caller decides.** Modules report what is seen (a gesture label, a
hand's position and confidence), never what it means. Turning a `stop` gesture into a command
is the job of the robot or agent consuming the results.

## What's in the box

| Concept | Public API | Role | Spec |
|---|---|---|---|
| Stream | `StreamProvider`, `Frame`, `FrameSource`, `OpenCVSource` | Single capture entry point: one thread reads a webcam, video file, URL, or injected source and publishes the newest frame | [specs/stream.md](specs/stream.md) |
| Pipeline | `Stage`, `Module`, `Result`, `Pipeline`, `LatestValue` | Staged-graph runtime: latest-value sampling, per-node threads and framerates, ordered start/stop | [specs/pipeline.md](specs/pipeline.md) |
| Hand | `HandStage`, `HandResult`, `Hand`, `HandDetector`, `MediaPipeHandDetector` | Shared stage: detects and crops hands **once per frame** for every hand-based module | [specs/hand.md](specs/hand.md) |
| Gesture classifier | `GestureClassifier`, `Gesture`, `HandGesture`, `ImageClassifier`, `HaGRIDViTClassifier`, `select_device` | Perception module: names the gesture in each hand crop (18 HaGRID classes) | [specs/gesture_classifier.md](specs/gesture_classifier.md) |
| Hand demo | `examples/hand_demo.py` | Runnable Gradio browser app wiring the whole graph end to end | [specs/hand_demo.md](specs/hand_demo.md) |

Everything in the table is implemented and exported from `vision_modules`.

## How it works

Every node samples the newest output of the node above it, processes it on its own thread,
and publishes its newest result for the nodes below. No queues, no backlog: a slow consumer
always sees the latest value and never falls behind, and a node never reprocesses a frame it
has already seen (stale skip on `frame_id`).

```
StreamProvider                 root: capture thread, publishes the latest Frame
  └─ HandStage                 shared stage: detect + crop once per frame (default 30 fps)
       └─ GestureClassifier    module: classifies each crop at its own rate (default 5 fps)
            (add more modules here, or another shared stage next to HandStage)
```

Key rules every node obeys:

- **Every published value carries `frame_id` and `ts`** from the originating `Frame`, so results
  from different modules can be matched on the same input frame.
- **`present=False` is a real result.** A module publishes one `Result` per processed frame;
  "no hand in this frame" is `present=False`, and only `None` means "not started yet".
- **A stage's real rate is capped by its upstream's.** `target_fps` is a ceiling, not a
  guarantee. `Stage.published_count` is the exact way to measure a stage's achieved rate.
- **Frames are shared by reference and read-only.** Anything that draws on a frame copies it
  first; anything that publishes a sub-region (the hand crop) copies it too.
- **Errors in `process()` never kill the graph.** They are recorded in `last_error`, logged,
  and the loop moves on to the next frame.
- **Threads, not processes.** OpenCV, NumPy, MediaPipe and PyTorch release the GIL during
  native work. Non-thread-safe resources (a MediaPipe instance, a loaded model) are created
  lazily on the worker thread and touched only there.

## Installation

Python 3.12 or newer.

```
pip install vision-modules            # core: numpy + opencv-python only
pip install "vision-modules[hand]"    # + hand stage and gesture classifier
```

Runtime dependencies are split by feature. The core package needs only `numpy` and
`opencv-python`. The `hand` extra adds `mediapipe` (hand detection) plus `torch`,
`transformers` and `pillow` (gesture classification). A future feature family (faces, poses)
gets its own extra, so you install only the perception you enable.

For development, `uv sync --dev` installs every extra plus the tooling and the demo's
dependencies.

## Quickstart

```python
import time

from vision_modules import GestureClassifier, HandStage, Pipeline, StreamProvider

provider = StreamProvider(0, mirror=True)   # webcam index 0, selfie view
hands = HandStage(provider, target_fps=30)  # detect + crop once per frame
gestures = GestureClassifier(hands, target_fps=5, threshold=0.55)

# Pipeline starts nodes upstream-first and stops them downstream-first.
with Pipeline([provider, hands, gestures]):
    seen = None
    while True:
        result = gestures.latest()
        if result is None or result.frame_id == seen:
            time.sleep(0.01)  # nothing new yet
            continue
        seen = result.frame_id
        hand = result.first
        if hand is not None and hand.label is not None:
            print(f"frame {result.frame_id}: {hand.label} ({hand.confidence:.2f})")
```

The first run downloads the models (see [Models and caches](#models-and-caches)).

`StreamProvider` also accepts a file path or URL (`StreamProvider("clip.mp4")`) or any object
implementing the `FrameSource` protocol, so the same graph runs over recordings or frames
pushed from elsewhere. `provider.ended` turns `True` when a file source runs out.

## API overview

### Stream

```
StreamProvider(source: int | str | FrameSource = 0, *, mirror: bool = False)
  .start() -> StreamProvider   # opens the source, spawns the capture thread; idempotent
  .latest() -> Frame | None    # newest frame, or None before the first read
  .ended -> bool               # True once the source reported no more frames
  .stop() -> None              # stops and joins the thread, then closes the source
  # also a context manager
```

`Frame` is a frozen dataclass: `frame_id` (monotonic, from 1), `ts` (`time.monotonic()` at
capture), `image` (`(H, W, 3)` `uint8` **BGR**, OpenCV's native order). BGR is the in-library
convention; consumers convert as needed and get a new array.

`FrameSource` is the seam for injecting frames without a camera:

```python
class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...   # next BGR image, or None when exhausted
    def close(self) -> None: ...
```

### Pipeline

```
Stage[TIn, TOut](upstream, target_fps: float | None, name: str | None = None)
  .start() / .stop() / .latest()
  .process(item) -> TOut | None   # subclass hook, runs on the worker thread only
  .close()                        # subclass hook, releases owned resources
  .target_fps                     # plain attribute, can be changed while running
  .published_count                # exact number of publishes so far
  .last_error                     # last exception raised by process(), if any

Module[TIn, TOut: Result](Stage)  # a Stage whose output is a perception Result
Result: frame_id, ts, present     # frozen base class of every published value
Pipeline(nodes).start() / .stop() # ordered lifecycle, also a context manager
```

`target_fps=None` means "as fast as the upstream delivers". Setting it at or above the
upstream's rate sees every item; setting it lower deliberately down-samples.

### Hand

```
HandStage(provider, target_fps=30, pad=0.35, max_hands=1, detector: HandDetector | None = None)
```

Publishes a `HandResult` per frame, also when no hand is found (`present=False`):

- `hands: tuple[Hand, ...]` in detector order, highest score first
- `first: Hand | None`

Each `Hand` has `bbox` (`(x0, y0, x1, y1)` in full-frame pixels, padded by `pad` and
clamped), `crop` (an independent BGR copy of that box, or `None` for a zero-area box at the
frame edge), and `score`.

`HandDetector` is a boxes-only protocol (`detect(image_bgr, ts) -> tuple[DetectedHand, ...]`),
so any detector can sit behind the stage. The shipped `MediaPipeHandDetector` uses the
MediaPipe Tasks `HandLandmarker` in `VIDEO` mode, taking the box as the extremes of the 21
landmarks. Landmarks and handedness stay internal for now.

### Gesture classifier

```
GestureClassifier(hand_stage, target_fps=5, threshold=0.55,
                  classifier: ImageClassifier | None = None, device: str | None = None)
```

Publishes a `Gesture` per processed `HandResult`, with `hands: tuple[HandGesture, ...]`
index-aligned with `HandResult.hands` and a `first` convenience. Each `HandGesture` has:

- `label: str | None`, the top class when its score clears `threshold`, else `None`
- `confidence: float`, the top score
- `scores: dict[str, float]`, every class's score, so you can apply your own rule

Labels are HaGRID's 18 class names, verbatim and never remapped:

`call`, `dislike`, `fist`, `four`, `like`, `mute`, `ok`, `one`, `palm`, `peace`,
`peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`,
`two_up_inverted`

The runtime authority is the classifier's `labels` property, so a fine-tuned backend with
another vocabulary works unchanged. `ImageClassifier` is the seam (`labels`,
`classify(image_bgr) -> dict[str, float]`, `close()`); the shipped `HaGRIDViTClassifier`
wraps the Hugging Face model `dima806/hand_gestures_image_detection`, a ViT fine-tuned on
HaGRID.

`threshold` and `target_fps` are plain attributes you can retune while the module runs.

### Writing your own module

Subclass `Module`, pick an upstream, implement `process()`, and return a frozen `Result`
subclass carrying the upstream's `frame_id` and `ts`:

```python
from dataclasses import dataclass

from vision_modules import HandResult, Module, Result


@dataclass(frozen=True)
class HandHeight(Result):
    y_center: float | None


class HandHeightModule(Module[HandResult, HandHeight]):
    def process(self, item: HandResult) -> HandHeight:
        hand = item.first
        if hand is None:
            return HandHeight(item.frame_id, item.ts, present=False, y_center=None)
        _, y0, _, y1 = hand.bbox
        return HandHeight(item.frame_id, item.ts, present=True, y_center=(y0 + y1) / 2)
```

Add it to the `Pipeline` list after its upstream and read it with `latest()` like any other
node. Heavy resources go in a lazily-initialized attribute used only inside `process()`, and
are released in `close()`.

## Models and caches

Nothing is bundled in the wheel. Models download on first use and are cached afterwards:

| Model | Used by | Size | Cache |
|---|---|---|---|
| MediaPipe `hand_landmarker.task` | `MediaPipeHandDetector` | a few MB | `~/.cache/vision-modules/`, overridable with `VISION_MODULES_CACHE` or `model_path=` |
| `dima806/hand_gestures_image_detection` | `HaGRIDViTClassifier` | ~340 MB | Hugging Face's own cache |

Inference device: `select_device()` picks `mps` if available, else `cuda`, else `cpu`. Pass
`device=` to override. The library never sets `PYTORCH_ENABLE_MPS_FALLBACK`; export it
yourself if an MPS op is missing.

## Demo

A Gradio browser app runs the hand stage and gesture classifier live over your webcam:

```
uv run python examples/hand_demo.py --mirror
```

Open the printed URL and grant the browser camera access. Flags: `--mirror`, `--hand-fps`
(default 30), `--classifier-fps` (default 5), `--threshold` (default 0.55).

The page shows the raw feed, the same frame with a box per detected hand, and the first hand's
full per-class score breakdown with the validated label marked. Sliders retune the threshold
and both stages' target fps live, and a readout shows each stage's real achieved fps next to
its target. The demo needs the `demo` dependency group, which `uv sync --dev` installs; it is
not part of the package.

## Development

This project is built **spec-first**: every concept is designed in `specs/` before it is
built, and each spec carries a status (`Draft`, `Stable`, `Implemented`, `Updated`) that
tracks whether the code matches it. Implementation plans in `plans/` turn settled specs into
buildable steps. Start with [AGENTS.md](AGENTS.md), then [specs/_index.md](specs/_index.md)
and [plans/_index.md](plans/_index.md).

```
uv sync --dev
uv run ruff check .
uv run ruff format .
uv run pyright
uv run pytest
```

Lint, type check, and tests must all pass before a change is done. `pyright` covers the tests
too.

### Tests

Two tiers, separated by directory:

| Tier | Directory | Network | Runs by default |
|---|---|---|---|
| Unit / integration | `tests/` | never | yes |
| Live / e2e | `tests-e2e/` | real models and devices | no |

`tests/` mirrors the `src/vision_modules/` layout and drives every stage through its seam
with scripted sources, detectors and classifiers, so no camera, model, or network is needed.
`tests/test_project_map.py` guards the docs: every module must appear in the AGENTS.md project
map and be governed by a spec whose frontmatter paths exist.

The live tier runs the real MediaPipe and ViT models against photos in `tests-e2e/fixtures/`
(one `hand_<gesture>.jpg` per gesture, plus `no_hand.jpg`) and asserts the full pipeline
predicts the photographed gesture:

```
uv run pytest tests-e2e
```

Tests that lack what they need skip instead of failing. Environment variables:

- `VISION_MODULES_CAMERA`: camera index for the webcam capture test (required for it; never
  guessed, because opening an unavailable camera can crash the process natively)
- `VISION_MODULES_HAND_IMAGE`: overrides the bundled hand photo
- `VISION_MODULES_CACHE`: where the MediaPipe model bundle is cached
