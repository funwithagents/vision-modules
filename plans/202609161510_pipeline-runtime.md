# Pipeline runtime

**Status:** Done

Implements `specs/pipeline.md` in full: the latest-value slot, the `Result` base, the generic `Stage` worker, the `Module` alias, and the `Pipeline` lifecycle helper. Delivers `src/vision_modules/pipeline.py` and its tests. It deliberately builds **no concrete stage** — the stream, hand and classifier plans build those on top of this.

## Context you need before starting

- Read `AGENTS.md`, `specs/pipeline.md` (all of it — it is the "what"; this file is the "how"), `specs/testing.md`.
- Prerequisite: plan `202609161500_dependency-extras.md` is `Done` (so `numpy` is installed; this plan itself uses only the standard library plus `numpy` in tests).
- `src/vision_modules/pipeline.py` exists as a docstring-only stub. Replace its body; keep a module docstring.
- Python is 3.12: use the PEP 695 generic syntax (`class Stage[TIn, TOut]:`), `X | None` unions, `from __future__ import annotations` is not needed.
- Threads must always be **joined** on `stop()`; a test that leaves a thread running will hang or flake the suite.

## Scope

- `src/vision_modules/pipeline.py` — everything below.
- `tests/test_pipeline.py` — functional tests, fake upstreams, no sleeps longer than needed.
- `specs/pipeline.md` — add `tests/test_pipeline.py` under `tests:` in the frontmatter (in the same step that creates the file — the drift test requires listed paths to exist); status flip `Stable` → `Implemented` at the end, in the file and in `specs/_index.md`.
- `plans/_index.md` + this file — status flips.

## Target API (exact names; the tests and later plans depend on them)

```python
import logging, threading, time
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

class LatestValue[T]:
    def __init__(self) -> None: ...            # holds a threading.Lock and a value slot
    def publish(self, value: T) -> None: ...   # replace the slot, under the lock
    def get(self) -> T | None: ...             # current value, under the lock

class HasFrameId(Protocol):                    # anything a Stage can consume
    @property
    def frame_id(self) -> int: ...

class Upstream[T](Protocol):                   # anything a Stage can sample
    def latest(self) -> T | None: ...

@dataclass(frozen=True)
class Result:
    frame_id: int
    ts: float
    present: bool

class Stage[TIn: HasFrameId, TOut: HasFrameId]:
    def __init__(self, upstream: Upstream[TIn], target_fps: float | None, name: str | None = None) -> None: ...
    name: str                       # given name, else type(self).__name__
    target_fps: float | None
    last_error: BaseException | None   # None until process() raises
    def start(self) -> "Stage[TIn, TOut]": ...
    def stop(self) -> None: ...
    def latest(self) -> TOut | None: ...
    def process(self, item: TIn) -> TOut | None: raise NotImplementedError   # subclass hook, worker thread only
    def close(self) -> None: ...    # subclass hook, default no-op, worker thread, once, as the loop exits
    def __enter__ / __exit__       # start() / stop()

class Module[TIn: HasFrameId, TOut: Result](Stage[TIn, TOut]):
    """A Stage whose output is a perception Result. No extra behaviour."""

class Node(Protocol):               # what Pipeline manages
    def start(self) -> object: ...
    def stop(self) -> None: ...

class Pipeline:
    def __init__(self, nodes: Sequence[Node]) -> None: ...
    def start(self) -> "Pipeline": ...   # nodes[0] first ... nodes[-1] last
    def stop(self) -> None: ...          # nodes[-1] first ... nodes[0] last; idempotent
    def __enter__ / __exit__
```

### Worker loop (write it exactly like this)

```
_last_id = None
while not stop_event.is_set():
    t0 = time.monotonic()
    item = upstream.latest()
    if item is not None and item.frame_id != _last_id:
        _last_id = item.frame_id
        try:
            out = self.process(item)
        except Exception as exc:                # noqa: BLE001 — spec: errors never kill the graph
            self.last_error = exc
            log.exception("%s: process() failed on frame %s", self.name, item.frame_id)
        else:
            if out is not None:
                self._slot.publish(out)
    # pacing
    if target_fps is None:
        stop_event.wait(0.001)               # tiny idle sleep; never busy-spin
    else:
        remaining = (1.0 / target_fps) - (time.monotonic() - t0)
        stop_event.wait(max(remaining, 0.0)) # wait() returns early when stop() fires
self.close()
```

Notes: `stop_event.wait(...)` instead of `time.sleep` is what makes `stop()` return promptly. `ruff` will flag the blind `except Exception` (rule BLE001); keep the `# noqa: BLE001` with the reason.

- `start()`: if already running, return `self`; else create the `threading.Thread(target=self._run, name=self.name, daemon=True)` and start it.
- `stop()`: set the event; if the thread exists, `join()` with **no timeout**; forget the thread so `stop()` is idempotent and `start()` can be called again later.
- `Pipeline.stop()` must stop every node even if one `stop()` raises (use `try/finally` or collect and re-raise after).

## Steps

1. Mark this plan `In progress` (file + `plans/_index.md`).
2. Write `src/vision_modules/pipeline.py` per the API above. Keep it one file, top-level docstring: one paragraph pointing at `specs/pipeline.md`.
3. Create `tests/test_pipeline.py` with the tests listed below, and add `  - tests/test_pipeline.py` under `tests:` in the frontmatter of `specs/pipeline.md`.
4. Run the verification gate; fix until green.
5. Flip `specs/pipeline.md` to `**Status:** Implemented` and its row in `specs/_index.md`; flip this plan to `Done` (file + index).

## Tests to write (`tests/test_pipeline.py`)

Test helpers, at the top of the file:

- `@dataclass(frozen=True) class Item: frame_id: int` — a minimal `HasFrameId`.
- `class FakeUpstream: def __init__(self): self.item = None; def latest(self): return self.item` — the test sets `.item`.
- `class RecordingStage(Stage[Item, Result])`: `process()` appends the item to `self.seen` and returns `Result(item.frame_id, 0.0, True)`; `close()` records `threading.get_ident()` into `self.closed_on`.
- `def wait_until(pred, timeout=2.0)`: poll `pred()` every 5 ms; `assert pred()` at the end with a message. Use it instead of fixed sleeps everywhere.

Tests (each is one function; the name says what it pins):

1. `test_latest_value_starts_empty_and_returns_newest` — `get()` is `None`; after `publish(1); publish(2)`, `get() == 2`.
2. `test_result_is_frozen` — assigning to a `Result` field raises `dataclasses.FrozenInstanceError`.
3. `test_stage_processes_each_new_frame_once` — upstream item `Item(1)`; start stage with `target_fps=None`; `wait_until(lambda: st.latest() is not None)`; then `wait_until` 50 ms passes (use `time.sleep(0.05)` once here, it's the point of the test); assert `st.seen == [Item(1)]` (processed once despite hundreds of loop iterations). Set `Item(2)`; `wait_until(lambda: len(st.seen) == 2)`; `st.latest().frame_id == 2`. Stop.
4. `test_stage_ignores_empty_upstream` — upstream item stays `None`; start, sleep 30 ms, `st.seen == []`, `st.latest() is None`. Stop.
5. `test_stage_target_fps_bounds_the_rate` — upstream whose `latest()` returns a fresh `Item` with an incrementing id on every call; `target_fps=20`; run 0.5 s; assert `5 <= len(st.seen) <= 15` (generous bounds — CI machines are noisy). Stop.
6. `test_process_returning_none_publishes_nothing` — a stage whose `process` returns `None`; after it has seen the item (`wait_until(lambda: st.seen)`), `st.latest() is None`.
7. `test_process_error_is_recorded_and_stage_keeps_going` — a stage whose `process` raises `ValueError` on `frame_id == 1` and returns a `Result` otherwise; feed `Item(1)` then `Item(2)`; `wait_until(lambda: st.latest() is not None)`; `isinstance(st.last_error, ValueError)`; `st.latest().frame_id == 2`.
8. `test_stop_joins_thread_and_calls_close_on_worker` — start, capture the worker via `threading.enumerate()` names or record `threading.get_ident()` inside `process`; stop; assert `st.closed_on == worker_ident` and `worker_ident != threading.get_ident()`, and no thread named `st.name` is alive.
9. `test_start_and_stop_are_idempotent` — `start(); start()` creates one thread (count alive threads with `st.name`); `stop(); stop()` does not raise.
10. `test_context_manager_starts_and_stops` — `with RecordingStage(...) as st:` sees an item; after the block no thread of that name is alive.
11. `test_pipeline_starts_in_order_and_stops_in_reverse` — three fake nodes that append `("start", i)` / `("stop", i)` to a shared list; `Pipeline([n0, n1, n2]).start()` then `.stop()`; assert the list equals `[("start",0),("start",1),("start",2),("stop",2),("stop",1),("stop",0)]`.
12. `test_pipeline_stops_remaining_nodes_when_one_stop_raises` — middle node's `stop()` raises `RuntimeError`; `Pipeline.stop()` raises `RuntimeError` **and** the other two nodes still recorded `"stop"`.
13. `test_module_is_a_stage_with_result_output` — subclass `Module[Item, Result]`, run one item through, `isinstance(mod.latest(), Result)`.

Every test that starts a stage must stop it — use `try/finally` or the context manager, so a failing assertion never leaves a thread running.

## Verification

```
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

All green (pytest: the 4 existing tests plus the 13 above). Mark this plan `Done` only then, and only together with the `specs/pipeline.md` status flip.
