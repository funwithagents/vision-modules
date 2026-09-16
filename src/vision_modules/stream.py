"""Stream — single capture entry point publishing the latest frame.

Governed by specs/stream.md.
"""

import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

import cv2
import numpy as np

from vision_modules.pipeline import LatestValue


@dataclass(frozen=True, eq=False)
class Frame:
    frame_id: int
    ts: float
    image: np.ndarray


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class OpenCVSource:
    def __init__(self, source: int | str) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"cannot open video source {source!r}")

    def read(self) -> np.ndarray | None:
        ok, img = self._cap.read()
        return img if ok else None

    def close(self) -> None:
        self._cap.release()


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
        self._ended = False

    def start(self) -> Self:
        if self._thread is not None:
            return self
        if isinstance(self._source_arg, (int, str)):
            self._source = OpenCVSource(self._source_arg)
        else:
            self._source = self._source_arg
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

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        assert self._source is not None
        # Closing before join is what unblocks a source whose read() is
        # currently blocked (e.g. waiting on the next frame): closing makes
        # that call return, so the worker notices the stop and exits.
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
                self._ended = True
                break
            if self._mirror:
                img = cv2.flip(img, 1)
            self._frame_id += 1
            ts = time.monotonic()
            self._slot.publish(Frame(self._frame_id, ts, img))
