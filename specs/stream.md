---
code:
  - src/vision_modules/stream.py
tests:
---

# Stream

**Status:** Stable

## Purpose

The single entry point for video into the library. A `StreamProvider` owns one capture source (webcam, video file, or an injected source), reads it on its own background thread, and publishes the newest frame behind a lock so any number of downstream stages can sample it at their own rate without contending for the camera. Everything else in the library ([pipeline.md](pipeline.md), [hand.md](hand.md)) consumes frames from here and never touches the capture device.

## Decided

### Data shape

- `Frame` — frozen dataclass, published once and never mutated:
  - `frame_id: int` — monotonically increasing per provider, starting at 1. This is the key every downstream result carries, so outputs from different modules can be matched to the same input.
  - `ts: float` — capture timestamp, `time.monotonic()` taken right after the read. Monotonic (not wall-clock) so latency arithmetic inside the pipeline is immune to clock adjustments, and so it can feed MediaPipe's video mode, which requires strictly increasing timestamps ([hand.md](hand.md)).
  - `image: np.ndarray` — `(H, W, 3)` `uint8` **BGR** (OpenCV's native order). BGR is the canonical in-library format; any conversion (RGB for a model, grayscale, …) happens in the consumer and yields a new array.

### API

```
StreamProvider(source: int | str | FrameSource = 0, *, mirror: bool = False)
  .start() -> StreamProvider       # opens the source, spawns the capture thread; idempotent
  .latest() -> Frame | None        # newest frame, or None before the first read
  .ended -> bool                   # True once the source reported no more frames
  .stop() -> None                  # stops the thread, joins it, closes the source; idempotent
  # context manager: `with StreamProvider(0) as sp:` == start() / stop()
```

- `source` is an `int` (OpenCV camera index), a `str` (file path / URL, opened through OpenCV), or any object satisfying the `FrameSource` protocol. `int` / `str` are wrapped in the built-in `OpenCVSource`. The device is opened in `start()`, not in the constructor, so constructing a provider has no side effects.
- `mirror=True` flips every frame horizontally (`cv2.flip(img, 1)`, a new array) before publishing — the "selfie view" a webcam user expects. It is applied once here so every consumer (and the demo's display) sees the same orientation.
- `FrameSource` protocol — the seam that keeps the capture loop testable without a camera:

  ```
  class FrameSource(Protocol):
      def read(self) -> np.ndarray | None: ...   # next BGR image, or None when exhausted / failed
      def close(self) -> None: ...
  ```

  A test injects a source that yields synthetic arrays; the default `tests/` tier never opens a real device (see [testing.md](testing.md)). Only the `tests-e2e/` tier touches OpenCV capture.

### Capture loop contract

- **Single reader.** Only the provider's thread calls `source.read()`. No other code reads the device.
- Per iteration: `read()` → under the lock, rebind the published `Frame` (new `frame_id`, new `ts`, new `image`). The lock only guarantees a consistent `(frame_id, ts, image)` triple; it is never held during the read.
- `read()` returning `None` (end of file, camera unplugged) ends the loop: the last `Frame` stays available from `latest()`, and `ended` becomes `True`, so consumers can tell "no new frame yet" from "no more frames ever".
- `stop()` sets a stop event, **joins the thread without a timeout** (a read blocks for at most one frame), and only then closes the source — never release a device a thread may still be reading.

### Frame ownership (subtle — get right)

- **Frames are passed by reference, never copied, by the provider.** `read()` allocates a fresh array per call; the provider only rebinds and never mutates in place, so a consumer holding frame *N* stays valid as the provider advances.
- **Consumers treat `Frame.image` as read-only.** Read-only use is zero-copy and safe (`cv2.cvtColor`, `cv2.flip`, `PIL.Image.fromarray` all read only). In-place mutation corrupts every other holder — watch for draw ops (`cv2.putText`, `cv2.rectangle`), slice assignment (`img[:] = …`) and `dst=` arguments. Anything that draws copies first.
- **Slices are views.** `image[y0:y1, x0:x1]` aliases the shared buffer; anyone publishing a sub-region (e.g. the hand crop in [hand.md](hand.md)) must `.copy()` it.

## Open questions

1. **File-input pacing.** A video file reads as fast as the disk allows; should the provider optionally pace to the file's native FPS so a recording behaves like a live camera during development? Recommendation: `pace_to_source_fps: bool = True` for `str` sources, ignored for cameras. Deferrable.
2. **Resolution / capture settings.** Whether `OpenCVSource` exposes width / height / FPS requests (`cv2.CAP_PROP_*`) and what the default is. Recommendation: optional `width` / `height` kwargs, nothing forced by default. Deferrable.
3. **Wall-clock timestamp.** Whether `Frame` also carries `time.time()` for consumers correlating with external logs. Deferrable — add when a consumer needs it.
