# Stage.published_count

**Status:** Done

Implements the settled behavior in `specs/pipeline.md` ("`Stage` — the generic node"): a `published_count: int` attribute on `Stage`, incremented once per successful publish. Fixes a real bug surfaced in `examples/hand_demo.py`: its `FpsMeter` measured `Δframe_id / Δt` on a stage's output, which actually reflects the *upstream's* rate, not the stage's own — so lowering the hand-fps slider to 1 didn't move the displayed number, because `frame_id` kept advancing at the camera's rate regardless of how often `HandStage` itself ran. `published_count` is the only thing that tracks a stage's real, own throughput.

## Scope

- `src/vision_modules/pipeline.py` — `Stage.__init__` gains `self.published_count = 0`; `_run()` increments it alongside the existing `self._slot.publish(out)`.
- `tests/test_pipeline.py` — a `ClockDrivenUpstream` test helper (frame_id advances on wall-clock time, independent of how often `latest()` is polled, unlike the existing `IncrementingUpstream` which only advances when polled) plus three tests: counts increment once per publish, stay at 0 when `process()` returns `None`, and — the scenario that reproduces the bug — a down-sampled stage's `published_count` stays far below its last `frame_id` when a much faster upstream is driving it.
- `examples/hand_demo.py` — `FpsMeter.sample()` takes `stage.published_count` (a plain `int`) instead of `stage.latest().frame_id` (an `int | None`); no more need to unwrap `.latest()` for this.
- `specs/pipeline.md` — already updated in the same change that reported this: `published_count` added to the `Stage` surface, plus a bullet explaining why `frame_id` can't be used to measure a stage's own rate.
- `specs/hand_demo.md` — update the "Real vs. target fps" section to describe sampling `published_count` instead of `frame_id`.

## Steps

1. Mark this plan `In progress` (file + index).
2. `pipeline.py`: add `published_count`, increment it in `_run()`.
3. `test_pipeline.py`: add `ClockDrivenUpstream` and the three tests above.
4. `hand_demo.py`: update `FpsMeter`/`on_fps_tick` to sample `published_count`.
5. Verification gate; flip `specs/pipeline.md` back to `Implemented` (file + index) and this plan to `Done` (file + index).

## Verification

```
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv run python examples/hand_demo.py --help
```

All pass (47 tests, including the two new `published_count` tests and the updated `test_stage_target_fps_bounds_the_rate` neighbours). Also confirmed `test_published_count_reflects_this_stages_rate_not_upstream_frame_id` fails against the pre-fix `FpsMeter` reasoning (a fast `ClockDrivenUpstream` driving a down-sampled stage) and passes now, i.e. it reproduces the exact bug the user hit before the fix.
