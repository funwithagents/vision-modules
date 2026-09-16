"""Pipeline runtime: latest-value staged graph shared by every vision module.

Governed by specs/pipeline.md.
"""

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

log = logging.getLogger(__name__)


class LatestValue[T]:
    """A single lock-protected slot: newest published value wins, no queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None

    def publish(self, value: T) -> None:
        with self._lock:
            self._value = value

    def get(self) -> T | None:
        with self._lock:
            return self._value


class HasFrameId(Protocol):
    @property
    def frame_id(self) -> int: ...


class Upstream[T](Protocol):
    def latest(self) -> T | None: ...


@dataclass(frozen=True)
class Result:
    frame_id: int
    ts: float
    present: bool


class Stage[TIn: HasFrameId, TOut: HasFrameId]:
    """A node that samples its upstream's latest value on its own thread and publishes its own."""

    def __init__(
        self, upstream: Upstream[TIn], target_fps: float | None, name: str | None = None
    ) -> None:
        self._upstream = upstream
        self.target_fps = target_fps
        self.name = name if name is not None else type(self).__name__
        self.last_error: BaseException | None = None
        self._slot: LatestValue[TOut] = LatestValue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        if self._thread is not None:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def latest(self) -> TOut | None:
        return self._slot.get()

    def process(self, item: TIn) -> TOut | None:
        raise NotImplementedError

    def close(self) -> None:
        """Subclass hook, worker thread only, called once as the loop exits."""

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
        _last_id: int | None = None
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            item = self._upstream.latest()
            if item is not None and item.frame_id != _last_id:
                _last_id = item.frame_id
                try:
                    out = self.process(item)
                except Exception as exc:  # spec: errors never kill the graph
                    self.last_error = exc
                    log.exception(
                        "%s: process() failed on frame %s", self.name, item.frame_id
                    )
                else:
                    if out is not None:
                        self._slot.publish(out)
            # pacing
            if self.target_fps is None:
                self._stop_event.wait(0.001)  # tiny idle sleep; never busy-spin
            else:
                remaining = (1.0 / self.target_fps) - (time.monotonic() - t0)
                self._stop_event.wait(
                    max(remaining, 0.0)
                )  # wait() returns early when stop() fires
        self.close()


class Module[TIn: HasFrameId, TOut: Result](Stage[TIn, TOut]):
    """A Stage whose output is a perception Result. No extra behaviour."""


class Node(Protocol):
    def start(self) -> object: ...
    def stop(self) -> None: ...


class Pipeline:
    """Starts nodes upstream-first, stops them downstream-first."""

    def __init__(self, nodes: Sequence[Node]) -> None:
        self._nodes = list(nodes)

    def start(self) -> Self:
        for node in self._nodes:
            node.start()
        return self

    def stop(self) -> None:
        errors: list[BaseException] = []
        for node in reversed(self._nodes):
            try:
                node.stop()
            except Exception as exc:  # noqa: BLE001 — stop every remaining node even if one fails
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
