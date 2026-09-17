---
code:
  - examples/hand_demo.py
tests:
  - tests/test_hand_demo.py
---

# Hand demo

**Status:** Implemented

## Purpose

The runnable, end-to-end demo that proves the graph works: a browser app wiring `StreamProvider` → `HandStage` → `GestureClassifier` ([pipeline.md](pipeline.md), [stream.md](stream.md), [hand.md](hand.md), [gesture_classifier.md](gesture_classifier.md)) together and showing every module's latest result live. It is an **application**, not a library concept: it lives in `examples/`, outside `src/`, and owns all the meaning it attaches to results (e.g. drawing a bounding box) — the library itself only perceives ([pipeline.md](pipeline.md) "The library perceives; the caller decides").

## Decided

### A Gradio browser app, not a native window

The demo is `examples/hand_demo.py`, built on [Gradio](https://gradio.app)'s streaming webcam component (`gr.Image(sources=["webcam"], streaming=True)`) rather than a `cv2.imshow` window. The browser captures frames from the user's camera and streams them to a Python callback, which annotates a copy and returns it to a **separate** output image component — an input/output pair, not a loopback into the same component, so the raw feed and the detection overlay are both visible side by side at once.

This changes where capture happens, but not the pipeline: a small `PushFrameSource` class implements [stream.md](stream.md)'s `FrameSource` protocol (`open()` / `read()` / `close()` / `fps()` — the last returns `None`, browser-pushed frames have no nominal rate), fed by `push(image_bgr)` calls from the streaming callback instead of `OpenCVSource` reading a local device. `read()` blocks until the next push; `close()` wakes it and makes it return `None` (that is how `StreamProvider.stop()` interrupts it), and `open()` clears the closed flag so the provider can be restarted. `StreamProvider` wraps it exactly as it would wrap a camera, so mirroring, frame-id assignment, and the rest of `StreamProvider`'s behaviour are unchanged. `HandStage` and `GestureClassifier` are constructed and started/stopped the normal way via `Pipeline`, wrapped around `demo.launch()`:

```python
with Pipeline([provider, hands, gestures]):
    demo.launch()
```

- **No native-GUI thread constraint applies here.** [pipeline.md](pipeline.md)'s concurrency rule ("every `cv2.imshow` / `cv2.waitKey` call must happen on the main thread") is specific to `cv2` windows; a Gradio app has no such window, so the streaming callback (which does the drawing) can run on whatever thread Gradio invokes it on. The lifecycle and stale-skip rules in [pipeline.md](pipeline.md) are otherwise unchanged — worker threads still only compute and publish.
- **Browser picks the camera, not the CLI.** There is no `--source` flag: the browser's own webcam picker chooses the device (and could offer several), so there is no server-side notion of a camera index or file path. A pre-recorded video file is out of scope for now (see Open questions).
- **BGR/RGB conversion happens at the boundary.** Gradio delivers and expects RGB `uint8` arrays; the library's internal convention (following `cv2`) is BGR. The callback converts both ways, once per frame.
- **The demo copies the frame before drawing on it** ([stream.md](stream.md) frame-ownership rules) — `canvas = frame.image.copy()`, never draw on `frame.image`.

### Layout: `gr.Blocks`, not `gr.Interface`

Classification text drawn on the video (the original design) was hard to read over a moving image, so results are shown as a separate component instead, and the layout needs more than one input/output component — both push the demo onto `gr.Blocks` rather than `gr.Interface`. Top to bottom:

1. **Three `gr.Slider`s in a `gr.Row`**, above the columns below — threshold (0–1, default `--threshold`), hand stage target fps and classifier target fps (defaults `--hand-fps`/`--classifier-fps`, both sharing one range, 1–`max(30, --hand-fps, --classifier-fps)`, so the two are visually and numerically comparable on the same scale). Each `.change()` handler writes straight to a mutable attribute the worker thread reads fresh every loop iteration — `gestures.threshold`, `hands.target_fps`, `gestures.target_fps` — so every one of them retunes the running pipeline live, no restart. (This is exactly [pipeline.md](pipeline.md)'s `Stage.target_fps`, a settable attribute the worker re-reads every iteration and never caches — nothing pipeline-side had to change to make it live-adjustable. It rejects non-positive values, which the sliders' floor of 1 never produces.)
   - The hand-fps slider only raises the *ceiling* `HandStage` computes against; it can't get more frames out of the browser than `stream_every` (fixed at launch, see below) delivers, so raising it past the launch value won't raise the achieved rate — the fps readout below shows the real, capped number either way.
2. **Snapshot row** (`gr.Row`): a `gr.Textbox` holding the snapshot folder, pre-filled with the directory of `hand_demo.py` itself (`DEMO_DIR = Path(__file__).resolve().parent`), then two `gr.Button`s — **Save detector snapshot** and **Save classifier snapshot** — and a `gr.Markdown` status line under the row. See "Snapshots" below.
3. **Three columns in a `gr.Row`:**
   - **Input** — `gr.Image(sources=["webcam"], streaming=True)`, the raw browser feed, untouched.
   - **Detected** — a second, output-only `gr.Image`: a copy of the same frame with only a green rectangle per detected hand drawn on it (`HandResult.hands[i].bbox`) — no text on the image itself.
   - **Classifications** — a single `gr.Label`, the first hand's full per-class score breakdown (`HandGesture.scores`). `gr.Label` sorts by score, so the top class is always the top row; when it also cleared `threshold` (`HandGesture.label` is not `None`), its key is prefixed with a checkmark so the validated call is visually distinguished from "just the highest score" without a second component. Placeholder single-entry dicts (`{"no hand": 1.0}`, `{"...": 1.0}`) stand in for "nothing to classify yet" / "hand present, classifier hasn't caught up".

Only the **first** hand's classification is shown in the panel (`Gesture.first`) — a v1 simplification for the demo's default single-hand case (see Open questions).

`cam_in.stream(fn=on_frame, inputs=cam_in, outputs=[cam_out, scores], stream_every=1 / args.hand_fps)` drives the loop; `on_frame` reads the raw frame from `cam_in` and returns `(annotated_frame, score_dict)` for the other two components. `stream_every` matters: it caps how often the browser is even allowed to hand the server a frame, independent of every `target_fps` downstream, and Gradio's own default (`0.5`, i.e. 2 fps) would silently bottleneck the whole pipeline below `--hand-fps` regardless of its value — so it's set from `--hand-fps` instead of left at the default.

### Snapshots

Each button saves what its module **last consumed** — `save_input` from [snapshot.md](snapshot.md) over `Stage.last_input` ([pipeline.md](pipeline.md)) — so the detector button writes the full frame `HandStage` last ran detection on and the classifier button writes the crop(s) `GestureClassifier` last classified. It is deliberately the input, not the output: the classifier publishes labels only, and re-reading `hands.latest()` at click time would usually give a newer crop than the one behind the label on screen.

- **File name:** `snapshot_<stage name>_<YYYYMMDDHHMMSS>.jpg` in the folder from the text box — `snapshot_HandStage_20260917030709.jpg`, and for the classifier `snapshot_GestureClassifier_20260917030709_0.jpg`: `save_crops` inserts the hand index before the suffix so the name stays index-aligned with `HandResult.hands` (one file per hand when `max_hands > 1`). The stage name is `Stage.name` (the class name by default); the time is wall-clock `datetime.now()` at the click, to the second — two clicks within the same second overwrite each other, an accepted limitation for a manual button. JPEG because these are quick visual snapshots, not a dataset; the library's `save_*` follow the suffix, so switching to `.png` is a one-character change.
- **Folder:** Gradio has no folder-picker component, so the folder is a plain path text box; missing directories are created by `save_input` itself. The default (the script's own directory) means a fresh checkout saves next to `hand_demo.py` without any setup.
- **Never raises into the UI.** `save_snapshot(stage, folder, now=None) -> str` wraps the call and returns a one-line status the `gr.Markdown` shows: the saved path(s) on success; a warning when the stage has not processed anything yet (`ValueError`), when the classifier's last input had no hand (an empty `HandResult` writes no file), or when the write fails (`OSError`). `snapshot_path(folder, stage_name, now) -> Path` is the pure naming helper; both take `now` explicitly so tests pin the file name.
- Runs on Gradio's request thread, reading `last_input` only — exactly the "caller thread, never a worker" rule of [snapshot.md](snapshot.md); the pipeline is untouched.

### Real vs. target fps

`target_fps` (`--hand-fps`, `--classifier-fps`) is a ceiling, not a guarantee: a stage's actual rate is capped by its upstream's actual rate ([pipeline.md](pipeline.md) "a stage's effective rate is capped by its upstream's"), and by how long its own `process()` takes. In this demo specifically, the achieved rate is also capped by `stream_every` (the browser→server delivery interval) and by real capture/network/inference latency, so a chosen `--hand-fps 30` is not a promise the pipeline hits 30.

The demo measures and displays the *actual* rate of every node, stream included, right under the sliders, e.g. `Stream: 28.9 fps (browser cap 30) · Hand stage: 24.3 fps (target 30) · Classifier: 4.8 fps (target 5)` — the "target" side reads `hands.target_fps`/`gestures.target_fps` live, so it tracks the fps sliders above rather than the CLI's original value. The stream has no `target_fps`; its ceiling is `stream_every` (fixed at launch from `--hand-fps`, see below), so that is the number shown next to it — the stream reading is the honest upstream cap the two stage readings are measured against. (`StreamProvider.source_fps` is `None` here: browser-pushed frames carry no device rate.)

- **`FpsMeter`** — holds only the last-seen count and timestamp for one stage. `sample(published_count)` returns `(published_count - last_count) / (now - last_t)`. This is exact, not an estimate — see below for why `published_count`, not `frame_id`.
- Driven by a `gr.Timer(1.0)` ticking once a second, independent of the webcam stream — so the fps readout keeps updating even if the video itself stalls, and its own 1 Hz cadence doesn't interfere with the measurement's correctness (see below).
- One `FpsMeter` per node (`provider`, `hands`, `gestures`); each samples that node's `published_count` directly ([stream.md](stream.md) gives the provider the same counter as a `Stage`) — no need to touch `.latest()` at all for this.

**Why `published_count`, not `frame_id`.** The first version of this sampled `stage.latest().frame_id`, on the reasoning that frame_id is strictly increasing and published once per item — true, but the wrong thing to measure: `frame_id` is inherited from the *originating* `Frame`, so it reflects which upstream item was processed, not how many times *this* stage has run. A down-sampled stage's published `frame_id` jumps ahead by however far the upstream advanced between two of its own cycles, so `Δframe_id / Δt` measures the upstream's rate leaking through, not the stage's own — this is why dragging the hand-fps slider down to 1 didn't move the displayed number at all, it stayed near the camera's actual capture rate. `Stage.published_count` ([pipeline.md](pipeline.md)) fixes this: it increments exactly once per successful publish and nothing else, so `Δpublished_count / Δt` is this stage's exact achieved rate regardless of what the upstream is doing.

### CLI

`--mirror`, `--hand-fps` (default 30), `--classifier-fps` (default 5), `--threshold` (default 0.55, the slider's starting value) — passed straight through to `StreamProvider`/`HandStage`/`GestureClassifier`. Defaults match [hand.md](hand.md) and [gesture_classifier.md](gesture_classifier.md).

### Dependency

`gradio` is a **dev-only `[dependency-groups]` entry** (`demo`, pulled into `dev`), not a `[project.optional-dependencies]` extra — no library consumer needs it to use `vision_modules`, only this repo's own example script does. See [project.md](project.md) ("Dev/example-only tooling"). It is imported inside `main()`, not at module level, so the module's pure helpers can be imported and tested without paying for the gradio import.

### Tests

The demo is an application, so it has no live test of its own, but its pure pieces — `PushFrameSource`, `FpsMeter`, `summarize`, `snapshot_path`, `save_snapshot` (driven through a real `HandStage` / `GestureClassifier` with scripted backends) — are unit-tested in `tests/test_hand_demo.py` (the `examples/` directory is on pytest's `pythonpath` and in pyright's scope, see [project.md](project.md)). The pipeline wiring it uses is covered end to end by `tests-e2e/test_hand_pipeline_live.py`.

## Open questions

1. **Video-file / non-browser input.** The original design supported a `--source` pointing at a camera index or a video file, read server-side via `OpenCVSource`. Dropped when the demo moved to a browser webcam; could come back as a second, `OpenCVSource`-backed code path if a file-driven demo is needed later. Deferrable.
2. **Turning perception into meaning.** The overlay only shows what was perceived (box, label, confidence). Attaching meaning to a label (e.g. a `stop` gesture triggering a visible action) is explicitly the caller's job per [pipeline.md](pipeline.md) and isn't built here. Deferrable — add if a concrete use case needs it.
3. **Multi-hand classification display.** The classifications panel only shows `Gesture.first`; with `max_hands > 1` the other hands' bounding boxes are still drawn on the "Detected" image, but their classifications aren't surfaced anywhere in the UI. Deferrable until a multi-hand use case actually needs it — would likely mean one panel per hand.
