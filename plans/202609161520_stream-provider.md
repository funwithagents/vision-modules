# Stream provider

**Status:** Done

Implements `specs/stream.md` in full: the `Frame` shape, the `FrameSource` seam, the `OpenCVSource` adapter, and the `StreamProvider` capture thread with its ownership rules. Delivers `src/vision_modules/stream.py`, its fast tests, and one opt-in live test. It builds **no downstream stage**.

## Context you need before starting

- Read `AGENTS.md`, `specs/stream.md` (all of it), `specs/testing.md` ("Live tier: skip without credentials").
- Prerequisites: `202609161500_dependency-extras.md` and `202609161510_pipeline-runtime.md` are `Done`. This plan reuses `LatestValue` from `vision_modules.pipeline`.
- `src/vision_modules/stream.py` is a docstring-only stub; replace its body.
- `opencv-python` is a **core** dependency, so `import cv2` at module top level is fine here.
- **Dataclasses that hold a NumPy array must be declared `@dataclass(frozen=True, eq=False)`.** With the default `eq=True`, comparing two instances compares arrays element-wise and raises "truth value of an array is ambiguous". This applies to `Frame` here and to `Hand` in the hand plan.

## Scope

- `src/vision_modules/stream.py` — everything below.
- `tests/test_stream.py` — fast tests with a fake `FrameSource`; never opens a camera.
- `tests-e2e/test_stream_live.py` — opens a real camera only when `VISION_MODULES_CAMERA` is set.
- `specs/stream.md` — frontmatter `tests:` gets both test paths; status `Stable` → `Implemented` at the end (file + `specs/_index.md`).
- `plans/_index.md` + this file — status flips.

## Target API (exact names)

```python
import threading, time
from dataclasses import dataclass
from typing import Protocol
import cv2
import numpy as np
from vision_modules.pipeline import LatestValue

@dataclass(frozen=True, eq=False)
class Frame:
    frame_id: int          # 1, 2, 3, ... per provider
    ts: float              # time.monotonic() right after the read
    image: np.ndarray      # (H, W, 3) uint8 BGR — treat as read-only

class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...   # next image, or None when exhausted / failed
    def close(self) -> None: ...

class OpenCVSource:
    def __init__(self, source: int | str) -> None: ...
        # cv2.VideoCapture(source); if not cap.isOpened(): cap.release(); raise RuntimeError(f"cannot open video source {source!r}")
    def read(self) -> np.ndarray | None: ...   # ok, img = cap.read(); return img if ok else None
    def close(self) -> None: ...               # cap.release()

class StreamProvider:
    def __init__(self, source: int | str | FrameSource = 0, *, mirror: bool = False) -> None: ...
        # stores the argument; opens NOTHING here
    def start(self) -> "StreamProvider": ...
    def latest(self) -> Frame | None: ...
    @property
    def ended(self) -> bool: ...
    def stop(self) -> None: ...
    def __enter__ / __exit__
```

### Behaviour to implement (from the spec)

- `start()`: idempotent. If `source` is an `int` or `str`, create `OpenCVSource(source)` **now** (so a bad camera index raises from `start()`, not from the constructor). Spawn `threading.Thread(target=self._run, name="StreamProvider", daemon=True)`.
- `_run()` loop, until the stop event is set: `img = source.read()`; if `img is None`: set `_ended = True` and break. If `mirror`: `img = cv2.flip(img, 1)` (a new array). `frame_id += 1`; `ts = time.monotonic()`; `slot.publish(Frame(frame_id, ts, img))`. **Never** write into `img` and never copy it.
- `stop()`: idempotent. Set the event; `join()` the thread with **no timeout**; then `source.close()`; forget the thread. Order matters: close only after the join.
- `ended` reads a `bool` set by the worker (a plain attribute is fine; the GIL makes a bool store atomic).
- `latest()` is just `slot.get()`.

## Steps

1. Mark this plan `In progress` (file + index).
2. Write `src/vision_modules/stream.py` per the API above.
3. Create `tests/test_stream.py` (below) and add `  - tests/test_stream.py` under `tests:` in `specs/stream.md`.
4. Create `tests-e2e/test_stream_live.py` (below) and add `  - tests-e2e/test_stream_live.py` under `tests:` in `specs/stream.md`.
5. Run the verification gate; fix until green. Also run `uv run pytest tests-e2e -q` once: with `VISION_MODULES_CAMERA` unset it must report the live test as **skipped**, not failed.
6. Flip `specs/stream.md` to `Implemented` (file + index); flip this plan to `Done` (file + index).

## Tests to write (`tests/test_stream.py`)

Test helper — a **steppable** fake source (put it at the top of the file):

```python
class SteppedSource:
    """Yields one image per allow(); read() blocks until allowed; close() unblocks it."""

    def __init__(self, images: list[np.ndarray]):
        self._images = list(images)
        self._gate = threading.Semaphore(0)
        self.closed = False
        self.reads = 0

    def allow(self, n: int = 1):
        [self._gate.release() for _ in range(n)]

    def read(self):
        self._gate.acquire()
        self.reads += 1
        if self.closed or not self._images:
            return None
        return self._images.pop(0)

    def close(self):
        self.closed = True
        self._gate.release()  # release a read blocked in acquire()
```

The `close()` release is essential: `StreamProvider.stop()` joins without a timeout, so a fake that stays blocked in `read()` would hang the test suite. Also reuse the `wait_until` helper pattern from `tests/test_pipeline.py` (copy it; test files don't import each other).

Images: build them with `np.zeros((4, 6, 3), np.uint8)` and put a distinct value somewhere (e.g. `img[:, 0] = 255` = white left column) so tests can tell frames and orientations apart.

1. `test_constructor_does_not_touch_the_source` — `StreamProvider(SteppedSource([...]))`; `src.reads == 0`; `latest() is None`; `ended is False`.
2. `test_frames_get_increasing_ids_and_monotonic_timestamps` — two images; start; `allow()`; `wait_until(lambda: sp.latest() is not None)`; `f1 = sp.latest()`; `f1.frame_id == 1`; `allow()`; `wait_until(lambda: sp.latest().frame_id == 2)`; `f2 = sp.latest()`; `f2.ts > f1.ts`; stop.
3. `test_frame_image_is_the_source_array_not_a_copy` — without mirror, `sp.latest().image is images[0]` (zero-copy by reference).
4. `test_earlier_frame_stays_valid_after_provider_advances` — keep `f1` from before the second `allow()`; after frame 2 is published, `f1.frame_id == 1` and `f1.image is images[0]` and `np.array_equal(f1.image, original_copy_of_images_0)` where you copied it before starting.
5. `test_mirror_flips_horizontally_into_a_new_array` — image with white **left** column, `mirror=True`; published image has the white column on the **right** (`image[:, -1].all() == 255` and `image[:, 0].all() == 0`), and `image is not images[0]`, and `images[0]` is unchanged.
6. `test_ended_becomes_true_when_source_is_exhausted` — one image; start; `allow(2)` (second read returns `None`); `wait_until(lambda: sp.ended)`; `sp.latest().frame_id == 1` (last frame stays available); stop.
7. `test_stop_joins_and_closes_the_source` — start, don't allow anything (worker is blocked in `read()`); `stop()` returns (within `wait_until`-scale time, i.e. the test simply doesn't hang); `src.closed is True`; no thread named `"StreamProvider"` alive. `stop()` again does not raise.
8. `test_context_manager_starts_and_stops` — `with StreamProvider(src) as sp:` → `allow()`, a frame arrives; after the block `src.closed`.
9. `test_int_source_that_cannot_open_raises_from_start` — `StreamProvider(source=999999)`; constructing is fine; `start()` raises `RuntimeError`. (OpenCV returns an unopened capture for a bogus index; if on your machine this is slow > 2 s, keep the test but note it in this plan's Verification.)

## Live test (`tests-e2e/test_stream_live.py`)

```python
from support import require_env   # tests-e2e/support.py — resolves because pytest puts tests-e2e/ on sys.path

def test_webcam_publishes_bgr_frames():
    index = int(require_env("VISION_MODULES_CAMERA"))     # skips when unset
    with StreamProvider(index) as sp:
        wait up to 5 s for sp.latest() to be non-None (poll every 50 ms), else fail with a message
        f = sp.latest()
        assert f.image.ndim == 3 and f.image.shape[2] == 3 and f.image.dtype == np.uint8
```

`from support import require_env` works as-is under `uv run pytest tests-e2e` (verified: pytest's default import mode puts the test file's directory on `sys.path`). Do not add any `sys.path` wiring to `tests-e2e/conftest.py`.

## Verification

```
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest                     # all fast tests green
uv run pytest tests-e2e -q        # live test reported as skipped when VISION_MODULES_CAMERA is unset
```

Then the status flips. Record here whether test 9 was slow on your machine.

Test 9 (`test_int_source_that_cannot_open_raises_from_start`) was not slow: the full 26-test fast suite (including this one) ran in 0.83s.

**Deviation from the API sketch above:** `stop()` calls `source.close()` *before* `thread.join()`, not after. Closing is what unblocks a source whose `read()` is currently blocked waiting for the next frame — for `SteppedSource` in the tests this is a semaphore acquire with no pending `allow()`, and the test suite hung indefinitely with the originally-specified join-then-close order (verified by running it). Closing first makes the blocked `read()` return, the worker notices the stop event and exits, and only then does `join()` return.
