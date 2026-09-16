"""Browser hand-gesture demo: Gradio's streaming webcam drives the pipeline.

Runnable application, not part of the `vision_modules` package. See
specs/hand_demo.md for the design.
"""

import argparse
import threading
import time

import cv2
import numpy as np

from vision_modules import (
    Gesture,
    GestureClassifier,
    HandResult,
    HandStage,
    LatestValue,
    Pipeline,
    StreamProvider,
)


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
    if hr is None:
        return
    for hand in hr.hands:
        x0, y0, x1, y1 = hand.bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)


def summarize(hr: HandResult | None, gesture: Gesture | None) -> dict[str, float]:
    """Per-class scores for gr.Label (which sorts by score, so the top entry is
    always the top class); the validated one, if any, is marked with a checkmark.
    """
    if hr is None or not hr.hands:
        return {"no hand": 1.0}
    first = gesture.first if gesture is not None else None
    if first is None or not first.scores:
        return {"...": 1.0}  # classifier hasn't caught up to the hand stage yet
    if first.label is None:
        return dict(first.scores)
    return {
        (f"✓ {label}" if label == first.label else label): score
        for label, score in first.scores.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser hand-gesture demo")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--hand-fps", type=float, default=30)
    parser.add_argument("--classifier-fps", type=float, default=5)
    parser.add_argument("--threshold", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    import gradio as gr  # only here, so the helpers above import without gradio

    args = parse_args()
    print("Note: the gesture classifier model may download on first run.")

    source = PushFrameSource()
    provider = StreamProvider(source, mirror=args.mirror)
    hands = HandStage(provider, target_fps=args.hand_fps)
    gestures = GestureClassifier(
        hands, target_fps=args.classifier_fps, threshold=args.threshold
    )

    def on_frame(frame_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        source.push(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        frame = provider.latest()
        if frame is None:
            return frame_rgb, {"no hand": 1.0}
        canvas = frame.image.copy()  # NEVER draw on frame.image
        hr = hands.latest()
        draw_boxes(canvas, hr)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), summarize(hr, gestures.latest())

    def on_threshold_change(value: float) -> None:
        gestures.threshold = value

    def on_hand_fps_change(value: float) -> None:
        hands.target_fps = value

    def on_classifier_fps_change(value: float) -> None:
        gestures.target_fps = value

    hand_fps_meter = FpsMeter()
    classifier_fps_meter = FpsMeter()

    def on_fps_tick() -> str:
        hand_fps = hand_fps_meter.sample(hands.published_count)
        classifier_fps = classifier_fps_meter.sample(gestures.published_count)
        hand_text = f"{hand_fps:.1f}" if hand_fps is not None else "—"
        classifier_text = f"{classifier_fps:.1f}" if classifier_fps is not None else "—"
        return (
            f"Hand stage: **{hand_text} fps** (target {hands.target_fps}) &nbsp;·&nbsp; "
            f"Classifier: **{classifier_text} fps** (target {gestures.target_fps})"
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
            with gr.Column():
                cam_in = gr.Image(sources=["webcam"], streaming=True, label="Input")
            with gr.Column():
                cam_out = gr.Image(label="Detected", interactive=False)
            with gr.Column():
                scores = gr.Label(label="Classifications")
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
            outputs=[cam_out, scores],
            stream_every=1 / args.hand_fps,
        )
        gr.Timer(1.0).tick(fn=on_fps_tick, outputs=fps_display)

    with Pipeline([provider, hands, gestures]):
        demo.launch()


if __name__ == "__main__":
    main()
