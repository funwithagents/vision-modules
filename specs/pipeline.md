---
code:
  - src/vision_modules/pipeline.py
tests:
  - tests/test_pipeline.py
---

# Pipeline

**Status:** Implemented

## Purpose

The runtime that lets several perception modules run at different framerates off a single stream while sharing upstream work. It is a **staged graph**: every node samples the newest output of the node above it, processes it on its own thread at its own rate, and publishes its newest result for the nodes below. This one pattern ("latest-value sampling") is what the whole library is built from: [stream.md](stream.md) is the root, [hand.md](hand.md) is a shared intermediate stage, and each perception module ([gesture_classifier.md](gesture_classifier.md) today) is a leaf. This spec defines the pattern, the base classes, the uniform module interface, and the concurrency rules every node obeys.

## Decided

### Shape of the graph

The runtime itself is graph-agnostic — `pipeline.py` knows nothing about hands, streams, or gestures; it only knows `Stage`/`Module`/`Pipeline`. Application code wires concrete nodes into whatever tree it needs, and a `Stage` can feed more than one downstream consumer. Today's example instantiation (built in `examples/hand_demo.py`, see [hand_demo.md](hand_demo.md)) is:

```
StreamProvider                 (root: capture thread, publishes the latest Frame)
  └─ HandStage                 (shared stage: detect + crop ONCE per frame)
       └─ GestureClassifier    (module, own framerate)
            (add more modules here, e.g. a FaceStage alongside HandStage, or another
             module reading HandStage/GestureClassifier's latest())
```

- Shared work is done once, in a shared stage, and fanned out. Modules never re-run upstream work (never their own hand detection, for instance). This replaces the rejected "one independent pipeline per module" shape, which duplicated detection.
- A node is both a consumer (of its upstream's `latest()`) and a provider (to its downstream's). That dual role is what gives every node an independent framerate.
- **The library perceives; the caller decides.** Modules report what is seen (a gesture label, a hand's position), never what it *means* (a "stop" command, a number to act on). Mapping perception to actions belongs to the robot / agent consuming the results, or to the demo.

### Latest-value sampling

- **`LatestValue[T]`** — one lock-protected slot. `publish(value)` replaces the slot; `get() -> T | None` returns the current value. Single writer, any number of readers. No queue, no backlog: a slow consumer sees the newest value and never falls behind.
- **Stale skip.** Every consumer remembers the `frame_id` of the last item it processed and skips when `latest()` returns the same one. A consumer that out-paces its upstream idles instead of reprocessing.
- **Every published value carries `frame_id` and `ts`** from the originating `Frame`, so results from different modules can be matched ("fused") on the same input frame by the consumer.

### `Stage` — the generic node

```
Stage[TIn, TOut](upstream, target_fps: float | None, name: str | None = None)
  .start() -> Stage       # spawns the worker thread; idempotent while running; restartable after stop()
  .stop() -> None         # signals and joins the thread (close() runs on the worker as it exits); idempotent
  .latest() -> TOut | None
  .process(item: TIn) -> TOut | None   # subclass hook; runs on the worker thread only
  .close() -> None                     # subclass hook; release owned resources, runs once per run
  .name: str              # defaults to the class name; for display / logging
  .target_fps: float | None   # settable while running; None = as fast as upstream; must be > 0
  .last_error: BaseException | None
  .published_count: int   # how many times this stage has actually published, ever (across restarts)
```

Worker loop: `item = upstream.latest()`; if `item` is `None` or its `frame_id` equals the last processed one, wait; else `out = self.process(item)`, publish `out` when not `None` and increment `published_count`, record the `frame_id`; then sleep so iterations respect `target_fps` (`None` = as fast as upstream delivers). Timing uses `time.monotonic()`.

- A stage's **effective rate is capped by its upstream's**: it can't publish fresher than it's fed. Set `target_fps` at or above the upstream's to see every item; lower it to deliberately down-sample.
- **`published_count` measures this stage's own real throughput; `frame_id` does not.** `frame_id` is inherited from the originating `Frame` and only reflects *whose* item was processed, not *how many* items this stage has processed — a down-sampled stage's published `frame_id` jumps ahead by however far the upstream advanced between two of *this* stage's cycles, so `Δframe_id / Δt` measures the upstream's rate, not this stage's own. `published_count` increments exactly once per successful `process()`+publish and nothing else, so `Δpublished_count / Δt` between any two samples (however far apart) is this stage's exact achieved rate — this is the only reliable way for a caller to check a stage's real fps against its `target_fps`.
- **Owned resources live on the worker thread.** Anything non-thread-safe a stage holds (a MediaPipe instance, a loaded model) is created lazily in the worker and touched only there. `close()` runs on the worker as it exits.
- **Owned vs. borrowed backends.** A stage that accepts an injected backend (`HandStage(detector=…)`, `GestureClassifier(classifier=…)`) treats it as **borrowed**: it uses it but never closes it — the caller created it, keeps it open across restarts, and closes it when done. A backend the stage created itself (the default MediaPipe detector, the default HaGRID classifier) is **owned**: `close()` releases it and drops the reference, so the next run creates a fresh one. This is what lets a stage be restarted with either kind of backend.
- **`target_fps` is validated.** It is a settable attribute (the demo retunes it live) but the worker divides by it, so a non-positive value is rejected with `ValueError` at construction and on assignment rather than killing the worker thread silently. `None` stays the "as fast as upstream" value.
- **Errors in `process` don't kill the graph.** The exception is recorded in `last_error` and logged; the loop continues with the next item (see open question 1).

### `Module` — the uniform perception interface

A **module** is a `Stage` whose output is a perception `Result` a robot / agent can act on. The uniform interface every module exposes — the property [_index.md](_index.md) promises — is exactly the `Stage` surface: `start()`, `stop()`, `latest()`, `name`, `target_fps`. Enabling a capability means constructing another module on the shared stage and starting it; nothing else in the graph changes.

- **`Result`** — base for every published value, frozen dataclass: `frame_id: int`, `ts: float`, `present: bool`. `present=False` is the graceful "nothing to report on this frame" (no hand, no face, …), so consumers always get a result per processed frame and only special-case `None` as "not started yet".
- Module outputs are typed subclasses (`Gesture`, …), never tuples or dicts. The brainstorm's `result_sink: dict` design is dropped in favour of `latest()` so a module is read exactly like every other node.
- **Per-subject tuples.** A module built on a shared stage that can find several subjects (hands today, faces or people later) reports one entry per subject, index-aligned with the stage's own tuple, plus a `first` convenience property. Supporting more subjects is then a `max_*` knob, never a breaking change to the result shape. See [hand.md](hand.md).

### Lifecycle

- Start upstream-first (provider → shared stage → modules) so nothing samples a node that isn't running; stop **downstream-first** (modules → shared stage → provider) so a node is never left reading from a closed upstream. A `Pipeline` helper owning an ordered list of nodes and doing this is in scope (`Pipeline(nodes).start() / .stop()`, context manager).
- `stop()` is bounded: worker loops check a stop event each iteration, so `stop()` returns within roughly one `target_fps` period plus one `process` call. Threads are joined, never abandoned.
- **Every node is restartable.** `start()` after `stop()` spawns a fresh worker: the stale-skip memory is per run (so a restarted stage treats the upstream's current value as new and processes it once), `close()` has already released the owned backend so the new run creates its own, and `latest()`, `published_count` and `last_error` carry over from the previous run. `StreamProvider` follows the same rule by re-opening its source ([stream.md](stream.md)). A `Pipeline` can therefore be started and stopped any number of times.
- **`Pipeline.start()` is all-or-nothing.** If a node's `start()` raises (an unopenable camera, say), the nodes already started are stopped again in reverse order and the original exception propagates — no half-started graph is left running behind a failed `with Pipeline(...)`. `Pipeline.stop()` stops every node even if one of them raises, then re-raises the first error.

### Concurrency rules (requirements)

- **Threads, not processes, not asyncio.** OpenCV, NumPy, MediaPipe and PyTorch release the GIL during native work, so threads give real overlap for these workloads.
- **Worker threads compute and publish only.** They never call GUI functions — on macOS every `cv2.imshow` / `cv2.waitKey` / window call must happen on the main thread. Display is the caller's job, on the main thread, from `latest()` values, after copying any frame it draws on ([stream.md](stream.md) ownership rules).
- **One non-thread-safe resource, one thread** (see `Stage` above).
- **Stale skip everywhere** — never reprocess an unchanged `frame_id`.

### Demo lives in `examples/`, not in the library

A runnable end-to-end demo lives at `examples/hand_demo.py`, outside `src/`, as an application built on this pipeline rather than part of it — specced on its own in [hand_demo.md](hand_demo.md).

### Non-goals (escalation paths, not built)

Latest-value + threads is the design. Only if a real need appears: a bounded `maxsize=1` drop-oldest queue for a consumer that must see *every* frame (recording, frame-accurate tracking); `multiprocessing` with `shared_memory` for heavy pure-Python modules or fault isolation; GStreamer (`tee` / `queue` / `videorate`) for production multi-branch streaming. None are in scope now.

## Open questions

1. **Error policy.** Record-and-continue (decided above) vs fail-fast per stage. Recommendation: keep record-and-continue; add `on_error="continue" | "stop"` later if needed. Deferrable.
2. **Consumer-side delivery.** Polling `latest()` is decided; should modules also offer a push path (callback on publish, or an `async` iterator) for agents that don't want to poll? Recommendation: add `subscribe(callback)` on `LatestValue` in a later revision; polling is enough for the first modules. Deferrable, but it shapes how a robot / agent integrates.
3. **Wake-up mechanism.** The worker sleeps to `target_fps` and re-polls; a `threading.Condition` notified on publish would cut latency for `target_fps=None` stages. Deferrable (optimization).
4. **Fusion.** Whether the library offers a helper to join results by `frame_id` across modules (e.g. compare two modules' outputs on the same frame). Recommendation: consumer-side for now; revisit once two modules run together. Deferrable.
