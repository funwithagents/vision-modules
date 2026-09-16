---
code:
  - src/vision_modules/pipeline.py
tests:
---

# Pipeline

**Status:** Stable

## Purpose

The runtime that lets several perception modules run at different framerates off a single stream while sharing upstream work. It is a **staged graph**: every node samples the newest output of the node above it, processes it on its own thread at its own rate, and publishes its newest result for the nodes below. This one pattern ("latest-value sampling") is what the whole library is built from: [stream.md](stream.md) is the root, [hand.md](hand.md) is a shared intermediate stage, and each perception module ([gesture_classifier.md](gesture_classifier.md) today) is a leaf. This spec defines the pattern, the base classes, the uniform module interface, and the concurrency rules every node obeys.

## Decided

### Shape of the graph

```
StreamProvider                 (root: capture thread, publishes the latest Frame)
  └─ HandStage                 (shared stage: detect + crop + landmarks ONCE per frame)
       └─ GestureClassifier    (module, own framerate)
            (add more modules here)
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
  .start() -> Stage       # spawns the worker thread; idempotent
  .stop() -> None         # signals, joins the thread, then close(); idempotent
  .latest() -> TOut | None
  .process(item: TIn) -> TOut | None   # subclass hook; runs on the worker thread only
  .close() -> None                     # subclass hook; release owned resources
  .name: str              # defaults to the class name; for display / logging
  .last_error: BaseException | None
```

Worker loop: `item = upstream.latest()`; if `item` is `None` or its `frame_id` equals the last processed one, wait; else `out = self.process(item)`, publish `out` when not `None`, record the `frame_id`; then sleep so iterations respect `target_fps` (`None` = as fast as upstream delivers). Timing uses `time.monotonic()`.

- A stage's **effective rate is capped by its upstream's**: it can't publish fresher than it's fed. Set `target_fps` at or above the upstream's to see every item; lower it to deliberately down-sample.
- **Owned resources live on the worker thread.** Anything non-thread-safe a stage holds (a MediaPipe instance, a loaded model) is created lazily in the worker and touched only there. `close()` runs on the worker as it exits.
- **Errors in `process` don't kill the graph.** The exception is recorded in `last_error` and logged; the loop continues with the next item (see open question 1).

### `Module` — the uniform perception interface

A **module** is a `Stage` whose output is a perception `Result` a robot / agent can act on. The uniform interface every module exposes — the property [_index.md](_index.md) promises — is exactly the `Stage` surface: `start()`, `stop()`, `latest()`, `name`, `target_fps`. Enabling a capability means constructing another module on the shared stage and starting it; nothing else in the graph changes.

- **`Result`** — base for every published value, frozen dataclass: `frame_id: int`, `ts: float`, `present: bool`. `present=False` is the graceful "nothing to report on this frame" (no hand, no face, …), so consumers always get a result per processed frame and only special-case `None` as "not started yet".
- Module outputs are typed subclasses (`Gesture`, …), never tuples or dicts. The brainstorm's `result_sink: dict` design is dropped in favour of `latest()` so a module is read exactly like every other node.
- **Per-subject tuples.** A module built on a shared stage that can find several subjects (hands today, faces or people later) reports one entry per subject, index-aligned with the stage's own tuple, plus a `first` convenience property. Supporting more subjects is then a `max_*` knob, never a breaking change to the result shape. See [hand.md](hand.md).

### Lifecycle

- Start upstream-first (provider → shared stage → modules) so nothing samples a node that isn't running; stop **downstream-first** (modules → shared stage → provider) so a node is never left reading from a closed upstream. A `Pipeline` helper owning an ordered list of nodes and doing this is in scope (`Pipeline(nodes).start() / .stop()`, context manager).
- `stop()` is bounded: worker loops check a stop event each iteration, so `stop()` returns within roughly one `target_fps` period plus one `process` call. Threads are joined, never abandoned.

### Concurrency rules (requirements)

- **Threads, not processes, not asyncio.** OpenCV, NumPy, MediaPipe and PyTorch release the GIL during native work, so threads give real overlap for these workloads.
- **Worker threads compute and publish only.** They never call GUI functions — on macOS every `cv2.imshow` / `cv2.waitKey` / window call must happen on the main thread. Display is the caller's job, on the main thread, from `latest()` values, after copying any frame it draws on ([stream.md](stream.md) ownership rules).
- **One non-thread-safe resource, one thread** (see `Stage` above).
- **Stale skip everywhere** — never reprocess an unchanged `frame_id`.

### Demo lives in `examples/`, not in the library

The runnable webcam demo (main-thread OpenCV window, overlays of every module's latest result, ESC to quit) is an `examples/hand_demo.py` script at the repo root, outside `src/`. It is an application: it owns the display loop, the platform (main-thread) constraint, and any meaning it attaches to results (e.g. turning a `stop` label into a red banner). It is built by the plan that first wires a module onto the hand stage, and the `examples/` directory is added to the [AGENTS.md](../AGENTS.md) project map when it appears. The demo copies the frame before drawing on it.

### Non-goals (escalation paths, not built)

Latest-value + threads is the design. Only if a real need appears: a bounded `maxsize=1` drop-oldest queue for a consumer that must see *every* frame (recording, frame-accurate tracking); `multiprocessing` with `shared_memory` for heavy pure-Python modules or fault isolation; GStreamer (`tee` / `queue` / `videorate`) for production multi-branch streaming. None are in scope now.

## Open questions

1. **Error policy.** Record-and-continue (decided above) vs fail-fast per stage. Recommendation: keep record-and-continue; add `on_error="continue" | "stop"` later if needed. Deferrable.
2. **Consumer-side delivery.** Polling `latest()` is decided; should modules also offer a push path (callback on publish, or an `async` iterator) for agents that don't want to poll? Recommendation: add `subscribe(callback)` on `LatestValue` in a later revision; polling is enough for the first modules. Deferrable, but it shapes how a robot / agent integrates.
3. **Wake-up mechanism.** The worker sleeps to `target_fps` and re-polls; a `threading.Condition` notified on publish would cut latency for `target_fps=None` stages. Deferrable (optimization).
4. **Fusion.** Whether the library offers a helper to join results by `frame_id` across modules (e.g. compare two modules' outputs on the same frame). Recommendation: consumer-side for now; revisit once two modules run together. Deferrable.
