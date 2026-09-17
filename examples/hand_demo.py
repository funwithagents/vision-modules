"""Browser hand-gesture demo: Gradio's streaming webcam drives the pipeline.

Runnable application, not part of the `vision_modules` package. See
specs/hand_demo.md for the design.
"""

import argparse
import threading
import time
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_modules import (
    Gesture,
    GestureClassifier,
    HandResult,
    HandStage,
    LatestValue,
    Pipeline,
    Stage,
    StreamProvider,
    save_input,
)

DEMO_DIR = Path(__file__).resolve().parent  # default snapshot folder

# One colour per hand slot (BGR, for cv2). A slot's panel label names the same colour.
SLOT_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),  # green
    (255, 128, 0),  # blue
    (0, 165, 255),  # orange
    (255, 0, 255),  # magenta
)
SLOT_NAMES: tuple[str, ...] = ("green", "blue", "orange", "magenta")


def slot_color(index: int) -> tuple[int, int, int]:
    return SLOT_COLORS[index % len(SLOT_COLORS)]


def slot_name(index: int) -> str:
    return SLOT_NAMES[index % len(SLOT_NAMES)]


class PushFrameSource:
    """FrameSource fed by frames pushed from the Gradio streaming callback.

    read() blocks until the next push; close() wakes it and makes it return None
    (how StreamProvider.stop() interrupts it); open() clears the closed flag so
    the provider can be restarted.
    """

    def __init__(self) -> None:
        self._slot: LatestValue[np.ndarray] = LatestValue()
        self._new_frame = threading.Event()
        self._closed = True  # until open()

    def open(self) -> None:
        self._closed = False
        self._new_frame.clear()

    def push(self, image_bgr: np.ndarray) -> None:
        self._slot.publish(image_bgr)
        self._new_frame.set()

    def read(self) -> np.ndarray | None:
        while not self._closed:
            if self._new_frame.wait(timeout=0.1):
                self._new_frame.clear()
                return self._slot.get()
        return None

    def close(self) -> None:
        self._closed = True
        self._new_frame.set()

    def fps(self) -> float | None:
        return None  # browser-pushed frames carry no device rate


class FpsMeter:
    """Real achieved fps of a Stage, from its published_count alone.

    published_count increments exactly once per publish and nothing else, so
    (delta published_count / delta time) between any two samples is the exact
    achieved rate over that window — no need to poll continuously. frame_id
    would NOT work here: it's inherited from the upstream Frame, so it reflects
    the upstream's rate, not this stage's own (see specs/pipeline.md).
    """

    def __init__(self) -> None:
        self._last_count: int | None = None
        self._last_t: float | None = None

    def sample(self, published_count: int) -> float | None:
        now = time.monotonic()
        fps = None
        if self._last_count is not None and self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:
                fps = (published_count - self._last_count) / dt
        self._last_count, self._last_t = published_count, now
        return fps


def draw_boxes(canvas: np.ndarray, hr: HandResult | None) -> None:
    """One rectangle per published hand in its slot colour, with the slot number
    (1-based) tagged just inside the top-left corner so a box can be matched to its panel."""
    if hr is None:
        return
    for i, hand in enumerate(hr.hands):
        x0, y0, x1, y1 = hand.bbox
        color = slot_color(i)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            canvas,
            str(i + 1),
            (x0 + 4, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def summarize(
    hr: HandResult | None, gesture: Gesture | None, index: int = 0
) -> dict[str, float]:
    """Per-class scores of hand slot `index` for a gr.Label (which sorts by score,
    so the top entry is always the top class); the validated one, if any, is
    marked with a checkmark.
    """
    if hr is None or index >= len(hr.hands):
        return {"no hand": 1.0}
    entry = (
        gesture.hands[index]
        if gesture is not None and index < len(gesture.hands)
        else None
    )
    if entry is None or not entry.scores:
        return {"...": 1.0}  # classifier hasn't caught up to the hand stage yet
    if entry.label is None:
        return dict(entry.scores)
    return {
        (f"✓ {label}" if label == entry.label else label): score
        for label, score in entry.scores.items()
    }


def snapshot_path(folder: str | PathLike[str], stage_name: str, now: datetime) -> Path:
    """`<folder>/snapshot_<stage name>_<YYYYMMDDHHMMSS>.jpg` (local wall-clock time)."""
    return Path(folder) / f"snapshot_{stage_name}_{now:%Y%m%d%H%M%S}.jpg"


def save_snapshot(
    stage: Stage[Any, Any], folder: str | PathLike[str], now: datetime | None = None
) -> str:
    """Save what `stage` last consumed (`save_input`) into `folder`; return a
    one-line status for the UI instead of raising, so a click can never error out."""
    if now is None:
        now = datetime.now(tz=UTC).astimezone()  # aware, in the machine's local zone
    target = snapshot_path(folder, stage.name, now)
    try:
        written = save_input(stage, target)
    except (ValueError, TypeError, OSError) as exc:
        return f"⚠️ {stage.name}: {exc}"
    paths = (
        [written]
        if isinstance(written, Path)
        else [p for p in written if p is not None]
    )
    if not paths:
        return f"⚠️ {stage.name}: no crop to save (no hand in its last input)"
    return "Saved " + ", ".join(f"`{p}`" for p in paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser hand-gesture demo")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--hand-fps", type=float, default=30)
    parser.add_argument("--classifier-fps", type=float, default=5)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-hands", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    import gradio as gr  # only here, so the helpers above import without gradio

    args = parse_args()
    print("Note: the gesture classifier model may download on first run.")

    source = PushFrameSource()
    provider = StreamProvider(source, mirror=args.mirror)
    hands = HandStage(provider, target_fps=args.hand_fps, max_hands=args.max_hands)
    gestures = GestureClassifier(
        hands, target_fps=args.classifier_fps, threshold=args.threshold
    )

    def on_frame(frame_rgb: np.ndarray) -> tuple[Any, ...]:
        source.push(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        frame = provider.latest()
        if frame is None:
            return (frame_rgb, *([{"no hand": 1.0}] * args.max_hands))
        canvas = frame.image.copy()  # NEVER draw on frame.image
        hr = hands.latest()
        draw_boxes(canvas, hr)
        gesture = gestures.latest()
        panels = [summarize(hr, gesture, i) for i in range(args.max_hands)]
        return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), *panels)

    def on_threshold_change(value: float) -> None:
        gestures.threshold = value

    def on_hand_fps_change(value: float) -> None:
        hands.target_fps = value

    def on_classifier_fps_change(value: float) -> None:
        gestures.target_fps = value

    def on_save_detector(folder: str) -> str:
        return save_snapshot(hands, folder)

    def on_save_classifier(folder: str) -> str:
        return save_snapshot(gestures, folder)

    stream_fps_meter = FpsMeter()
    hand_fps_meter = FpsMeter()
    classifier_fps_meter = FpsMeter()

    def fmt(fps: float | None) -> str:
        return f"{fps:.1f}" if fps is not None else "—"

    def on_fps_tick() -> str:
        stream_fps = stream_fps_meter.sample(provider.published_count)
        hand_fps = hand_fps_meter.sample(hands.published_count)
        classifier_fps = classifier_fps_meter.sample(gestures.published_count)
        # The stream has no target_fps: its ceiling is stream_every, fixed at
        # launch from --hand-fps (see cam_in.stream below).
        return (
            f"Stream: **{fmt(stream_fps)} fps** (browser cap {args.hand_fps:g}) &nbsp;·&nbsp; "
            f"Hand stage: **{fmt(hand_fps)} fps** (target {hands.target_fps}) &nbsp;·&nbsp; "
            f"Classifier: **{fmt(classifier_fps)} fps** (target {gestures.target_fps})"
        )

    with gr.Blocks(title="vision-modules hand demo") as demo:
        with gr.Row():
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=args.threshold,
                label="Gesture confidence threshold",
            )
            fps_max = max(30.0, args.hand_fps, args.classifier_fps)
            hand_fps = gr.Slider(
                minimum=1,
                maximum=fps_max,
                step=1,
                value=args.hand_fps,
                label="Hand stage target fps",
            )
            classifier_fps = gr.Slider(
                minimum=1,
                maximum=fps_max,
                step=1,
                value=args.classifier_fps,
                label="Classifier target fps",
            )
        fps_display = gr.Markdown("Hand stage: — fps &nbsp;·&nbsp; Classifier: — fps")
        with gr.Row():
            snapshot_folder = gr.Textbox(
                value=str(DEMO_DIR), label="Snapshot folder", scale=3
            )
            save_detector = gr.Button("Save detector snapshot", scale=1)
            save_classifier = gr.Button("Save classifier snapshot", scale=1)
        snapshot_status = gr.Markdown("")
        with gr.Row():
            # One row: two video columns (scale 2), then one narrow panel per slot
            # (scale 1). min_width is lowered from Gradio's 320px default, which
            # otherwise wraps the last column onto a new row at laptop widths.
            with gr.Column(scale=2, min_width=240):
                cam_in = gr.Image(sources=["webcam"], streaming=True, label="Input")
            with gr.Column(scale=2, min_width=240):
                cam_out = gr.Image(label="Detected", interactive=False)
            score_panels: list[Any] = []
            for i in range(args.max_hands):
                with gr.Column(scale=1, min_width=120):
                    score_panels.append(
                        gr.Label(
                            label=f"Hand {i + 1} ({slot_name(i)})", num_top_classes=3
                        )
                    )
        threshold.change(on_threshold_change, inputs=threshold, outputs=None)
        hand_fps.change(on_hand_fps_change, inputs=hand_fps, outputs=None)
        classifier_fps.change(
            on_classifier_fps_change, inputs=classifier_fps, outputs=None
        )
        # stream_every caps how often the browser can hand us a frame at all,
        # fixed at launch from the CLI's --hand-fps: raising the hand_fps
        # slider above that later can't get more frames than this allows (the
        # fps readout below will show the real, capped rate either way).
        cam_in.stream(
            fn=on_frame,
            inputs=cam_in,
            outputs=[cam_out, *score_panels],
            stream_every=1 / args.hand_fps,
        )
        gr.Timer(1.0).tick(fn=on_fps_tick, outputs=fps_display)
        save_detector.click(
            on_save_detector, inputs=snapshot_folder, outputs=snapshot_status
        )
        save_classifier.click(
            on_save_classifier, inputs=snapshot_folder, outputs=snapshot_status
        )

    with Pipeline([provider, hands, gestures]):
        demo.launch()


if __name__ == "__main__":
    main()
