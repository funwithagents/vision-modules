---
code:
  - src/vision_modules/snapshot.py
tests:
  - tests/test_snapshot.py
---

# Snapshot

**Status:** Implemented

## Purpose

A caller-side way to write to a local file the image a node is currently working with — either what it **publishes** (the newest full frame from a `StreamProvider`, the newest hand crops from a `HandStage`) or what it **consumed** (the exact input a module last handed to `process()`, e.g. the crop the gesture classifier last classified) — for debugging, dataset collection, or attaching evidence to a decision. The pipeline ([pipeline.md](pipeline.md)) is built on latest-value sampling: every node exposes its newest output through `latest()` and its newest input through `last_input`, and both are already safe to read from any thread without a copy ([stream.md](stream.md) frame-ownership rules). Saving is therefore a pure function *of a published value*, not a feature of the graph — it never touches a worker thread, never adds a queue, and never changes what a node publishes.

## Decided

### Which nodes have a "current frame"

Every value with pixels the library publishes is one of two types, and every stage's input is one of the values its upstream publishes — so the two views are covered by the same two writers:

| Node | Publishes (`latest()`) | Pixels in it | Consumes (`last_input`) | Pixels in it |
|---|---|---|---|---|
| `StreamProvider` | `Frame` | `image` — full BGR frame | — (root, no upstream) | — |
| `HandStage` | `HandResult` | `hands[i].crop` — one BGR copy per hand (`None` for a zero-area box) | `Frame` | the full frame it detected on |
| `GestureClassifier` | `Gesture` | none — labels and scores only | `HandResult` | the crop(s) it classified |

- **Output view** — `save_frame` writes a `Frame`, `save_crops` writes a `HandResult`. Pass `provider.latest()` / `hands.latest()` straight in.
- **Input view** — `save_input(stage, path)` reads `stage.last_input` and dispatches on its type to one of the two writers. This is how a module that publishes no pixels (`GestureClassifier`, any future label-only module) gets its "current frame": the input it last consumed, pinned by the stage itself rather than re-read from the upstream. Re-reading the upstream's `latest()` and matching `frame_id` is **not** equivalent — a 5 fps classifier under a 30 fps hand stage will find the upstream has moved on five times out of six ([pipeline.md](pipeline.md) "`last_input`").

No image field is added to `Gesture` for this: modules keep reporting perception only, and the shared stage keeps owning the pixels ([hand.md](hand.md)).

### API

```
save_frame(frame: Frame | None, path: str | PathLike[str]) -> Path
save_crops(result: HandResult | None, path: str | PathLike[str]) -> tuple[Path | None, ...]
save_input(stage: Stage, path: str | PathLike[str]) -> Path | tuple[Path | None, ...]
```

- **`save_frame`** writes `frame.image` to `path` and returns the path it wrote, as a `Path`. `frame` is typed `Frame | None` so a caller can pass `provider.latest()` straight in; `None` (nothing published yet) raises `ValueError` — silently writing nothing would hide the one state a caller most needs to notice.
- **`save_crops`** writes one file per hand in `result.hands`, and returns a tuple **index-aligned with `result.hands`** (the per-subject tuple rule of [pipeline.md](pipeline.md)): the written `Path` for a hand with a crop, `None` for a hand whose `crop` is `None`. `path` names the file for the first hand; hand *i* gets the index inserted before the suffix — `crop.png` → `crop_0.png`, `crop_1.png`, … — so the file names are stable whatever `max_hands` is. An empty result (`present=False`) writes nothing and returns `()`. `None` raises `ValueError`, as above.
- **Format follows the suffix**, via OpenCV's writer (`.png`, `.jpg`, `.bmp`, `.tiff`, …). Arrays are BGR — the library's canonical order ([stream.md](stream.md)) and exactly what `cv2.imwrite` expects — so **no color conversion happens anywhere in this module**; a PNG round-trips pixel-exact. An unsupported suffix raises `ValueError` before anything is touched on disk (checked with `cv2.haveImageWriter`).
- **Missing parent directories are created** (`mkdir(parents=True, exist_ok=True)`): a snapshot is a debugging aid and shouldn't fail because `out/` doesn't exist yet. A write that still fails (permissions, disk) raises `OSError` naming the path — `cv2.imwrite`'s `False` return is never swallowed.
- **`save_input`** writes `stage.last_input`: a `Frame` exactly as `save_frame` would (returning the `Path`), a `HandResult` exactly as `save_crops` would (returning the index-aligned tuple) — the return shape follows the stage's input type, which a caller wiring the graph knows. A stage that has not processed anything yet raises `ValueError`; an input type this module has no writer for (a custom stage consuming a custom value) raises `TypeError`. The dispatch is a closed `isinstance` chain over the library's own two pixel-bearing types — a third such type is added here, in the same change that introduces it.
- **Naming is the caller's.** The functions take a full file path; anyone saving a series builds names from the value's own key, e.g. `save_frame(f, out_dir / f"{f.frame_id:06d}.png")`. `frame_id` is unique per provider (monotonic across restarts, [stream.md](stream.md)) and is the same key every downstream result carries, so a frame file and a crop file named from the same `frame_id` are trivially matched.

### Invariants

**Consumer:** the hand demo's two snapshot buttons ([hand_demo.md](hand_demo.md) "Snapshots") are the first caller of `save_input`, one per module.


- **Read-only.** Writing never mutates or copies the source array beyond what the encoder needs; `Frame.image` stays shared and untouched, as the ownership rules require.
- **Caller thread, never a worker.** `save_*` is called from wherever the caller reads `latest()` / `last_input` — the main thread, a UI request handler, an agent loop. Workers "compute and publish only" ([pipeline.md](pipeline.md)); this module gives them nothing to do. It is not a stage and holds no state.
- **Snapshot, not recorder.** These save *the newest value at the moment of the call* — the newest output, or the input of the newest `process()` call. Latest-value sampling drops frames by design, so "save every frame" is not something this API can promise — that is the `maxsize=1` recording queue [pipeline.md](pipeline.md) lists as a non-goal.
- **No new dependencies.** OpenCV is already a core dependency ([project.md](project.md)); the module imports nothing from the `hand` extra beyond the `HandResult` type (`hand.py` itself imports MediaPipe lazily, so this costs nothing).

Exported from the package root like every other concept: `from vision_modules import save_frame, save_crops, save_input`.

## Open questions

1. **Sidecar metadata.** Whether to optionally write a `.json` next to the image with `frame_id`, `ts`, and (for crops) `bbox` and `score`, so a saved crop can be located in its frame later. Recommendation: add as an opt-in `metadata=True` flag once a dataset-collection use exists. Deferrable.
2. **Annotated snapshot.** Saving the frame *with* hand boxes drawn (the demo's "Detected" view). Recommendation: the caller draws on a copy and calls `cv2.imwrite` itself, or a `draw=` hook is added here later; the library stays perception-only. Deferrable.
