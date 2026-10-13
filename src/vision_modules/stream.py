"""Stream — single capture entry point publishing the latest frame.

Governed by specs/stream.md.
"""

import threading
import time
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import numpy as np

from vision_modules.pipeline import HasFrameId, LatestValue

if TYPE_CHECKING:
    import cv2

_OPENCV_HINT = 'install the opencv extra: pip install "vision-modules[opencv]"'


def _import_cv2(feature: str) -> ModuleType:
    """Import OpenCV on first use; it is an optional extra, not a core dependency."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError(f"{feature} requires OpenCV; {_OPENCV_HINT}") from e
    return cv2


@dataclass(frozen=True, eq=False)
class Frame:
    frame_id: int
    ts: float
    image: np.ndarray


@runtime_checkable
class FrameLike(HasFrameId, Protocol):
    """What a frame consumer reads: `frame_id`, `ts` and a (H, W, 3) uint8 BGR
    `image` treated as read-only. `Frame` is the concrete class StreamProvider
    publishes; stages type against this protocol so a host's own frame class,
    published from its own latest-value slot, is a valid upstream as it stands.
    Members are properties so a frozen dataclass field satisfies them."""

    @property
    def ts(self) -> float: ...
    @property
    def image(self) -> np.ndarray: ...


class FrameSource(Protocol):
    """A frame supplier whose lifecycle the provider drives: open() on every
    start(), close() on every stop(). close() may be called from another thread
    while read() is blocked and must make that read() return."""

    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...
    def fps(
        self,
    ) -> (
        float | None
    ): ...  # nominal rate the device reports; None when unknown / not open


class OpenCVSource:
    """cv2.VideoCapture behind the FrameSource protocol. read() and close() share
    a lock because VideoCapture is not thread-safe: release() waits for an
    in-flight read() (at most one frame period) instead of racing it."""

    def __init__(self, source: int | str) -> None:
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self._cv2: ModuleType | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        cv2 = _import_cv2("OpenCVSource")
        with self._lock:
            if self._cap is not None:
                return
            self._cv2 = cv2
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"cannot open video source {self._source!r}")
            self._cap = cap

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._cap is None:
                return None
            ok, img = self._cap.read()
            return img if ok else None

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def fps(self) -> float | None:
        # Same lock as read(): VideoCapture.get() is no more thread-safe than
        # read(). Many webcams report 0, which is "unknown", not a rate.
        with self._lock:
            if self._cap is None or self._cv2 is None:
                return None
            value = self._cap.get(self._cv2.CAP_PROP_FPS)
            return value if value > 0 else None


class StreamProvider:
    def __init__(
        self, source: int | str | FrameSource = 0, *, mirror: bool = False
    ) -> None:
        self._source_arg = source
        self._mirror = mirror
        self._source: FrameSource | None = None
        self._slot: LatestValue[Frame] = LatestValue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_id = 0
        self._last_ts = float("-inf")  # ts is strictly increasing per provider
        self._ended = False
        # Exact number of frames ever published (across restarts) — same
        # contract as Stage.published_count, so one meter fits every node.
        self.published_count = 0

    def start(self) -> Self:
        """Open the source and spawn the capture thread. No-op while running;
        after stop() it re-opens the source and starts a fresh run."""
        if self._thread is not None:
            return self
        if self._source is None:
            if isinstance(self._source_arg, (int, str)):
                self._source = OpenCVSource(self._source_arg)
            else:
                self._source = self._source_arg
        self._source.open()
        self._ended = False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="StreamProvider", daemon=True
        )
        self._thread.start()
        return self

    def latest(self) -> Frame | None:
        return self._slot.get()

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def source_fps(self) -> float | None:
        """Rate the source claims for itself; None before start(), after stop(),
        or when the source has no nominal rate (see published_count for the
        rate actually achieved)."""
        return self._source.fps() if self._source is not None else None

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        assert self._source is not None
        # Closing before join is what unblocks a source whose read() is
        # currently blocked (e.g. waiting on the next frame): closing makes
        # that call return, so the worker notices the stop and exits. The
        # FrameSource protocol requires close() to be safe to call while
        # read() is in progress on the worker.
        self._source.close()
        self._thread.join()
        self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def _run(self) -> None:
        assert self._source is not None
        while not self._stop_event.is_set():
            img = self._source.read()
            if img is None:
                if not self._stop_event.is_set():
                    self._ended = True  # the source ran out; a stop() is not an end
                break
            if self._mirror:
                img = np.ascontiguousarray(img[:, ::-1])  # new array, as cv2.flip
            self._frame_id += 1
            self.published_count += 1
            # Strictly after the previous frame's ts: on Windows before Python 3.13,
            # time.monotonic() ticks every ~15.6 ms, so two reads can share a reading.
            # Nudging keeps the host's clock, which perf_counter would not (stream.md).
            ts = max(time.monotonic(), self._last_ts + 1e-6)
            self._last_ts = ts
            self._slot.publish(Frame(self._frame_id, ts, img))
