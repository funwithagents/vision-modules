# Demo snapshot buttons

**Status:** Done

Implements the settled behavior in `specs/hand_demo.md` ("Snapshot row", "Snapshots"). Delivers a folder text box and two buttons in `examples/hand_demo.py` that save the detector's and the classifier's last input as `snapshot_<stage>_<YYYYMMDDHHMMSS>.jpg`, with a status line instead of exceptions, plus unit tests for the two pure helpers. Deliberately leaves out a real folder picker (Gradio has none), PNG/format selection, and annotated snapshots (`specs/snapshot.md` open question).

## Scope

- `examples/hand_demo.py` — `DEMO_DIR`, `snapshot_path`, `save_snapshot`; the snapshot `gr.Row` (textbox + two buttons) under the fps readout and a `gr.Markdown` status; two `.click` handlers.
- `tests/test_hand_demo.py` — tests for `snapshot_path`, `DEMO_DIR`, and `save_snapshot` through a real `HandStage` / `GestureClassifier` with scripted backends: frame and crop round-trip, "nothing processed" and "no hand" messages.
- `specs/hand_demo.md` — layout item and "Snapshots" section (done alongside); `specs/snapshot.md` — open question 2 closed, consumer noted.
- `README.md` — one sentence in "Demo".
- `specs/_index.md`, `plans/_index.md` — statuses.

## Steps

1. **Helpers.** `snapshot_path(folder, stage_name, now) -> Path` builds `Path(folder) / f"snapshot_{stage_name}_{now:%Y%m%d%H%M%S}.jpg"`. `save_snapshot(stage, folder, now=None) -> str` calls `save_input(stage, snapshot_path(...))`, catches `ValueError` / `TypeError` / `OSError` into a `⚠️` message, flattens the `Path | tuple` return, and reports "no crop to save" when nothing was written.
2. **Handlers.** `on_save_detector(folder)` → `save_snapshot(hands, folder)`; `on_save_classifier(folder)` → `save_snapshot(gestures, folder)`.
3. **UI.** After `fps_display`: a `gr.Row` with `gr.Textbox(value=str(DEMO_DIR), label="Snapshot folder", scale=3)` and the two buttons (`scale=1`), then `snapshot_status = gr.Markdown("")`. Wire `button.click(handler, inputs=snapshot_folder, outputs=snapshot_status)` next to the other event bindings.
4. **Tests.** Use a flat-colour image so the JPEG round-trip is pixel-exact; pin `now` to a fixed `datetime` and assert the exact file name and the status message.
5. **Docs and statuses.** README sentence; `specs/hand_demo.md` `Updated` → `Implemented` and this plan → `Done`, files and indexes, with the verified code.

## Verification

- `uv run pytest tests/test_hand_demo.py` passes.
- Full gate, all clean:

  ```
  uv run ruff check .
  uv run ruff format .
  uv run pyright
  uv run pytest
  ```

  Then mark this plan `Done` (here and in [_index.md](_index.md)) and the spec `Implemented`.
