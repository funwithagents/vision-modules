import dataclasses
import threading
import time
from dataclasses import dataclass

import pytest

from vision_modules.pipeline import (
    LatestValue,
    Module,
    Pipeline,
    Result,
    Stage,
    Upstream,
)

# --- Test helpers ------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    frame_id: int


class FakeUpstream:
    def __init__(self) -> None:
        self.item: Item | None = None

    def latest(self) -> Item | None:
        return self.item


class IncrementingUpstream:
    """Every call to latest() returns a fresh Item with the next id."""

    def __init__(self) -> None:
        self._next_id = 0

    def latest(self) -> Item:
        item = Item(self._next_id)
        self._next_id += 1
        return item


class ClockDrivenUpstream:
    """frame_id advances on wall-clock time, independent of how often latest()
    is polled — simulates a real producer (e.g. a camera thread) that keeps
    running regardless of how fast a downstream consumer samples it.
    """

    def __init__(self, fps: float) -> None:
        self._fps = fps
        self._start = time.monotonic()

    def latest(self) -> Item:
        return Item(int((time.monotonic() - self._start) * self._fps))


class RecordingStage(Stage[Item, Result]):
    def __init__(
        self,
        upstream: Upstream[Item],
        target_fps: float | None,
        name: str | None = None,
    ) -> None:
        super().__init__(upstream, target_fps, name)
        self.seen: list[Item] = []
        self.closed_on: int | None = None

    def process(self, item: Item) -> Result | None:
        self.seen.append(item)
        return Result(item.frame_id, 0.0, True)

    def close(self) -> None:
        self.closed_on = threading.get_ident()


class NoneStage(RecordingStage):
    def process(self, item: Item) -> Result | None:
        self.seen.append(item)
        return None


class FlakyStage(RecordingStage):
    def process(self, item: Item) -> Result | None:
        self.seen.append(item)
        if item.frame_id == 1:
            raise ValueError("boom")
        return Result(item.frame_id, 0.0, True)


def wait_until(pred: object, timeout: float = 2.0) -> None:
    assert callable(pred)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    assert pred(), "condition not met within timeout"


# --- LatestValue / Result -----------------------------------------------------


def test_latest_value_starts_empty_and_returns_newest() -> None:
    slot = LatestValue[int]()
    assert slot.get() is None
    slot.publish(1)
    slot.publish(2)
    assert slot.get() == 2


def test_result_is_frozen() -> None:
    result = Result(frame_id=1, ts=0.0, present=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.frame_id = 2  # type: ignore[misc]


# --- Stage ---------------------------------------------------------------------


def test_stage_processes_each_new_frame_once() -> None:
    upstream = FakeUpstream()
    upstream.item = Item(1)
    st = RecordingStage(upstream, target_fps=None)
    st.start()
    try:
        wait_until(lambda: st.latest() is not None)
        time.sleep(0.05)
        assert st.seen == [Item(1)]

        upstream.item = Item(2)
        wait_until(lambda: len(st.seen) == 2)
        result = st.latest()
        assert result is not None
        assert result.frame_id == 2
    finally:
        st.stop()


def test_stage_ignores_empty_upstream() -> None:
    upstream = FakeUpstream()
    st = RecordingStage(upstream, target_fps=None)
    st.start()
    try:
        time.sleep(0.03)
        assert st.seen == []
        assert st.latest() is None
    finally:
        st.stop()


def test_stage_target_fps_bounds_the_rate() -> None:
    upstream = IncrementingUpstream()
    st = RecordingStage(upstream, target_fps=20)
    st.start()
    try:
        time.sleep(0.5)
    finally:
        st.stop()
    assert 5 <= len(st.seen) <= 15


def test_published_count_increments_once_per_publish() -> None:
    upstream = FakeUpstream()
    upstream.item = Item(1)
    st = RecordingStage(upstream, target_fps=None)
    st.start()
    try:
        wait_until(lambda: st.published_count == 1)
        upstream.item = Item(2)
        wait_until(lambda: st.published_count == 2)
    finally:
        st.stop()
    assert st.published_count == len(st.seen) == 2


def test_published_count_reflects_this_stages_rate_not_upstream_frame_id() -> None:
    """A downsampled stage's frame_id can jump far ahead of how many times it
    actually ran — published_count is the only thing that tracks the latter.
    """
    upstream = ClockDrivenUpstream(fps=100)  # far faster than this stage's target
    st = RecordingStage(upstream, target_fps=10)
    st.start()
    try:
        time.sleep(0.5)
    finally:
        st.stop()
    last = st.latest()
    assert last is not None
    assert (
        last.frame_id > st.published_count
    )  # upstream ran far faster than we published
    assert 2 <= st.published_count <= 8  # roughly this stage's own ~10fps cap over 0.5s


def test_process_returning_none_publishes_nothing() -> None:
    upstream = FakeUpstream()
    upstream.item = Item(1)
    st = NoneStage(upstream, target_fps=None)
    st.start()
    try:
        wait_until(lambda: st.seen)
        assert st.latest() is None
        assert st.published_count == 0
    finally:
        st.stop()


def test_process_error_is_recorded_and_stage_keeps_going() -> None:
    upstream = FakeUpstream()
    upstream.item = Item(1)
    st = FlakyStage(upstream, target_fps=None)
    st.start()
    try:
        wait_until(lambda: st.last_error is not None)
        upstream.item = Item(2)
        wait_until(lambda: st.latest() is not None)
        assert isinstance(st.last_error, ValueError)
        result = st.latest()
        assert result is not None
        assert result.frame_id == 2
    finally:
        st.stop()


def test_stop_joins_thread_and_calls_close_on_worker() -> None:
    upstream = FakeUpstream()
    upstream.item = Item(1)
    st = RecordingStage(upstream, target_fps=None, name="worker-stage")
    st.start()
    try:
        wait_until(lambda: st.latest() is not None)
        worker = next(t for t in threading.enumerate() if t.name == "worker-stage")
        worker_ident = worker.ident
    finally:
        st.stop()
    assert st.closed_on == worker_ident
    assert worker_ident != threading.get_ident()
    assert not any(t.name == "worker-stage" for t in threading.enumerate())


def test_start_and_stop_are_idempotent() -> None:
    upstream = FakeUpstream()
    st = RecordingStage(upstream, target_fps=None, name="idempotent-stage")
    st.start()
    st.start()
    alive = [t for t in threading.enumerate() if t.name == "idempotent-stage"]
    assert len(alive) == 1
    st.stop()
    st.stop()


def test_context_manager_starts_and_stops() -> None:
    upstream = FakeUpstream()
    upstream.item = Item(1)
    with RecordingStage(upstream, target_fps=None, name="ctx-stage") as st:
        wait_until(lambda: st.latest() is not None)
    assert not any(t.name == "ctx-stage" for t in threading.enumerate())


def test_stage_restarts_after_stop_as_a_fresh_run() -> None:
    upstream = FakeUpstream()
    upstream.item = Item(1)
    st = RecordingStage(upstream, target_fps=None, name="restart-stage")
    st.start()
    wait_until(lambda: st.published_count == 1)
    st.stop()
    first_close = st.closed_on
    assert first_close is not None
    assert not any(t.name == "restart-stage" for t in threading.enumerate())

    st.closed_on = None
    st.start()
    try:
        # The new run has no stale-skip memory: the upstream's current item is
        # processed once more, and the counter carries over from the first run.
        wait_until(lambda: st.published_count == 2)
        time.sleep(0.05)
        assert st.seen == [Item(1), Item(1)]
        upstream.item = Item(2)
        wait_until(lambda: st.published_count == 3)
        result = st.latest()
        assert result is not None
        assert result.frame_id == 2
    finally:
        st.stop()
    assert st.closed_on is not None  # close() ran again as the second run exited
    assert not any(t.name == "restart-stage" for t in threading.enumerate())


def test_target_fps_rejects_non_positive_values() -> None:
    upstream = FakeUpstream()
    with pytest.raises(ValueError):
        RecordingStage(upstream, target_fps=0)
    st = RecordingStage(upstream, target_fps=10)
    with pytest.raises(ValueError):
        st.target_fps = -1
    assert st.target_fps == 10  # the bad assignment changed nothing
    st.target_fps = None
    assert st.target_fps is None


def test_target_fps_can_be_lowered_while_running() -> None:
    upstream = IncrementingUpstream()
    st = RecordingStage(upstream, target_fps=None)
    st.start()
    try:
        wait_until(lambda: len(st.seen) >= 20)
        st.target_fps = 10
        time.sleep(0.05)  # let the in-flight iteration drain
        before = len(st.seen)
        time.sleep(0.5)
        processed = len(st.seen) - before
    finally:
        st.stop()
    assert 2 <= processed <= 8  # ~10 fps over 0.5 s, nowhere near the unpaced rate


# --- Pipeline --------------------------------------------------------------------


class RecordingNode:
    def __init__(self, index: int, events: list[tuple[str, int]]) -> None:
        self._index = index
        self._events = events

    def start(self) -> "RecordingNode":
        self._events.append(("start", self._index))
        return self

    def stop(self) -> None:
        self._events.append(("stop", self._index))


class RaisingNode(RecordingNode):
    def stop(self) -> None:
        self._events.append(("stop", self._index))
        raise RuntimeError("boom")


class FailingStartNode(RecordingNode):
    def start(self) -> "RecordingNode":
        self._events.append(("start", self._index))
        raise RuntimeError("cannot open")


def test_pipeline_starts_in_order_and_stops_in_reverse() -> None:
    events: list[tuple[str, int]] = []
    nodes = [RecordingNode(i, events) for i in range(3)]
    pipeline = Pipeline(nodes)
    pipeline.start()
    pipeline.stop()
    assert events == [
        ("start", 0),
        ("start", 1),
        ("start", 2),
        ("stop", 2),
        ("stop", 1),
        ("stop", 0),
    ]


def test_pipeline_start_rolls_back_started_nodes_when_a_later_start_fails() -> None:
    events: list[tuple[str, int]] = []
    nodes = [
        RecordingNode(0, events),
        RecordingNode(1, events),
        FailingStartNode(2, events),
        RecordingNode(3, events),
    ]
    with pytest.raises(RuntimeError, match="cannot open"):
        Pipeline(nodes).start()
    assert events == [
        ("start", 0),
        ("start", 1),
        ("start", 2),
        ("stop", 1),
        ("stop", 0),
    ]


def test_pipeline_with_block_leaves_no_worker_running_when_start_fails() -> None:
    upstream = FakeUpstream()
    st = RecordingStage(upstream, target_fps=None, name="leak-stage")
    with pytest.raises(RuntimeError), Pipeline([st, FailingStartNode(1, [])]):
        pass  # never reached
    assert not any(t.name == "leak-stage" for t in threading.enumerate())


def test_pipeline_stops_remaining_nodes_when_one_stop_raises() -> None:
    events: list[tuple[str, int]] = []
    n0 = RecordingNode(0, events)
    n1 = RaisingNode(1, events)
    n2 = RecordingNode(2, events)
    pipeline = Pipeline([n0, n1, n2])
    pipeline.start()
    with pytest.raises(RuntimeError):
        pipeline.stop()
    assert ("stop", 0) in events
    assert ("stop", 2) in events


# --- Module -----------------------------------------------------------------------


class SimpleModule(Module[Item, Result]):
    def process(self, item: Item) -> Result | None:
        return Result(item.frame_id, 0.0, True)


def test_module_is_a_stage_with_result_output() -> None:
    upstream = FakeUpstream()
    upstream.item = Item(1)
    with SimpleModule(upstream, target_fps=None) as mod:
        wait_until(lambda: mod.latest() is not None)
        assert isinstance(mod.latest(), Result)
