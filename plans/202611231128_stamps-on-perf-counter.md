# Stamps on `time.perf_counter()`: `ts` and the stage timings

**Status:** Done

Implements the settled behavior in `specs/stream.md` ("Data shape", the `FrameLike` clock rule) and `specs/pipeline.md` (worker loop, "Stage timings": "One clock", "Ordering"). Moves every stamp the library publishes — `Frame.ts` from `StreamProvider`, `StageTiming.started_at` and `computed_at` — from `time.monotonic()` to `time.perf_counter()`, so they share a clock with a host that stamps its own frames with `perf_counter()` and measure durations finely on every platform. Deliberately leaves out any change to what the stamps mean, to `StageTiming` or any result's shape, to pacing (which stays on `time.monotonic()`), to the hand demo's meters, and to the deadlines in the tests.

**Compatibility.** On macOS, on Linux, and on Windows from Python 3.13, the two functions read the same clock: nothing observable changes. On Windows with Python 3.12 they have different origins, so a host there that compares `ts` with its own `time.monotonic()` readings must switch to `time.perf_counter()`. No such host is known. No version change.

## Scope

- `src/vision_modules/stream.py` — the capture loop reads `time.perf_counter()` for `ts`; the 1 µs strict-increase guard stays; comment reworded.
- `src/vision_modules/pipeline.py` — `_run` reads `time.perf_counter()` for `started_at` / `computed_at`; the pacing reads stay on `time.monotonic()`; a comment says which is which; `StageTiming` docstring.
- `tests/test_stream.py` — a published `Frame`'s `ts` lies between two `perf_counter()` readings; the frozen-clock test patches `perf_counter`.
- `tests/test_pipeline.py` — the bracketing test reads `perf_counter()` around and inside `process()`.
- `tests/test_hand.py`, `tests/test_gesture_classifier.py` — frames "on the stamps' clock" are stamped with `perf_counter()`; the first gap is asserted non-negative and small.
- `README.md` — the clock named beside `ts`, `StageTiming` and the face detector's `ts`.
- `specs/stream.md`, `specs/pipeline.md`, `specs/face.md`, `specs/_index.md` — the design; `stream.md` and `pipeline.md` go `Updated` while the code lags and back to `Implemented` when this plan is `Done`. `face.md` is an editorial edit (its example of a caller's stamp).
- `plans/_index.md` — this plan's row.

## Steps

1. **`ts`.** In `stream.py`, `ts = max(time.perf_counter(), self._last_ts + 1e-6)`. Reword the comment: the guard is no longer explained by a coarse tick.
2. **Stage stamps.** In `pipeline.py`, `started_at` and `computed_at` read `time.perf_counter()`. `t0` and `remaining` keep `time.monotonic()`. Add the comment distinguishing stamps from pacing reads; fix the `StageTiming` docstring.
3. **Stream tests.** New: `before = perf_counter()`, let the source yield one image, wait for the frame, `after = perf_counter()`, assert `before <= ts <= after`. Frozen-clock test: patch `perf_counter` in place of `monotonic`.
4. **Pipeline test.** The bracketing test (`before <= started_at <= inside <= computed_at <= after`) takes all three of its own readings from `perf_counter()`.
5. **Hand and chain tests.** Stamp test frames with `perf_counter()`; add `started_at - ts < 1.0` beside the existing `ts <= started_at`.
6. **Docs and statuses.** README; the two specs back to `Implemented` (file and index); this plan `Done`.

## Verification

- Steps 3 to 5 are the tests that fail on the Windows runner (Python 3.12) if a stamp is left on `time.monotonic()`; the pacing tests pass unchanged.
- `grep -rn "monotonic()" specs README.md src` names `time.monotonic()` only for pacing and for the comparison with `perf_counter()`.
- Then the full gate, all clean:

  ```
  uv run ruff check .
  uv run ruff format .
  uv run pyright
  uv run pytest
  ```
