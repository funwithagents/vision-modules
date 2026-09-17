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
        self._target_fps: float | None = None
        self.target_fps = target_fps
        self.name = name if name is not None else type(self).__name__
        self.last_error: BaseException | None = None
        self.published_count = 0
        self._slot: LatestValue[TOut] = LatestValue()
        self._input_slot: LatestValue[TIn] = LatestValue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def target_fps(self) -> float | None:
        """Rate ceiling; None = as fast as upstream delivers. Settable while running."""
        return self._target_fps

    @target_fps.setter
    def target_fps(self, value: float | None) -> None:
        if value is not None and value <= 0:
            raise ValueError(f"target_fps must be positive or None, got {value!r}")
        self._target_fps = value

    def start(self) -> Self:
        """Spawn the worker. No-op while running; after stop() it starts a fresh run."""
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

    @property
    def last_input(self) -> TIn | None:
        """The item most recently handed to process() — set before it runs, so it
        is the offending input while process() is failing; carries over restarts."""
        return self._input_slot.get()

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
                self._input_slot.publish(
                    item
                )  # a reference: published values are immutable
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
                        self.published_count += 1
            # pacing (read target_fps once: it can be reassigned from another thread)
            target_fps = self._target_fps
            if target_fps is None:
                self._stop_event.wait(0.001)  # tiny idle sleep; never busy-spin
            else:
                remaining = (1.0 / target_fps) - (time.monotonic() - t0)
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
        """All-or-nothing: if a node fails to start, the ones already started are stopped."""
        started: list[Node] = []
        for node in self._nodes:
            try:
                node.start()
            except BaseException:
                for exc in _stop_all(started):
                    log.warning("stopping a node after a failed start raised: %r", exc)
                raise
            started.append(node)
        return self

    def stop(self) -> None:
        errors = _stop_all(self._nodes)
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


def _stop_all(nodes: Sequence[Node]) -> list[BaseException]:
    """Stop every node, last first, even if some raise; return what was raised."""
    errors: list[BaseException] = []
    for node in reversed(nodes):
        try:
            node.stop()
        except Exception as exc:  # noqa: BLE001 — stop every remaining node even if one fails
            errors.append(exc)
    return errors
