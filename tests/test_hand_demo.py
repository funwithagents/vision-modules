"""Unit tests for the pure helpers of examples/hand_demo.py (on pytest's pythonpath)."""

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import hand_demo
import numpy as np
import pytest
from hand_demo import (
    DEMO_DIR,
    SLOT_COLORS,
    FpsMeter,
    PushFrameSource,
    draw_boxes,
    save_snapshot,
    snapshot_path,
    summarize,
)

from vision_modules import (
    DetectedHand,
    Frame,
    Gesture,
    GestureClassifier,
    Hand,
    HandGesture,
    HandResult,
    HandStage,
)

# --- FpsMeter ------------------------------------------------------------------


def test_fps_meter_needs_two_samples_then_reports_the_exact_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(hand_demo.time, "monotonic", lambda: clock[0])
    meter = FpsMeter()
    assert meter.sample(0) is None  # nothing to compare against yet
    clock[0] += 2.0
    assert meter.sample(10) == 5.0  # 10 publishes over 2 s
    clock[0] += 0.5
    assert meter.sample(10) == 0.0  # stalled stage reads 0, not None


# --- summarize -------------------------------------------------------------------


def _hand_result(n_hands: int) -> HandResult:
    hands = tuple(
        Hand((0, 0, 8, 8), np.zeros((8, 8, 3), np.uint8), 0.9) for _ in range(n_hands)
    )
    return HandResult(1, 1.0, present=bool(hands), hands=hands)


def _gesture(label: str | None, scores: dict[str, float]) -> Gesture:
    return Gesture(
        1, 1.0, present=True, hands=(HandGesture(label, max(scores.values()), scores),)
    )


def test_summarize_without_a_hand() -> None:
    assert summarize(None, None) == {"no hand": 1.0}
    assert summarize(_hand_result(0), _gesture("palm", {"palm": 1.0})) == {
        "no hand": 1.0
    }


def test_summarize_hand_seen_but_classifier_not_caught_up() -> None:
    assert summarize(_hand_result(1), None) == {"...": 1.0}
    empty = Gesture(1, 1.0, present=False, hands=())
    assert summarize(_hand_result(1), empty) == {"...": 1.0}


def test_summarize_below_threshold_shows_plain_scores() -> None:
    scores = {"fist": 0.4, "palm": 0.6}
    assert summarize(_hand_result(1), _gesture(None, scores)) == scores


def test_summarize_marks_the_validated_label() -> None:
    out = summarize(_hand_result(1), _gesture("palm", {"fist": 0.2, "palm": 0.8}))
    assert out == {"fist": 0.2, "✓ palm": 0.8}


def test_summarize_index_beyond_published_hands_is_no_hand() -> None:
    assert summarize(_hand_result(1), _gesture("palm", {"palm": 1.0}), index=1) == {
        "no hand": 1.0
    }


def test_summarize_index_with_hand_but_no_gesture_entry_is_pending() -> None:
    # two hands published, classifier result only has one entry so far
    assert summarize(_hand_result(2), _gesture("palm", {"palm": 1.0}), index=1) == {
        "...": 1.0
    }


def test_summarize_reads_the_requested_slot() -> None:
    two = Gesture(
        1,
        1.0,
        present=True,
        hands=(
            HandGesture("palm", 0.8, {"fist": 0.2, "palm": 0.8}),
            HandGesture("fist", 0.7, {"fist": 0.7, "palm": 0.3}),
        ),
    )
    assert summarize(_hand_result(2), two, index=0) == {"fist": 0.2, "✓ palm": 0.8}
    assert summarize(_hand_result(2), two, index=1) == {"✓ fist": 0.7, "palm": 0.3}


# --- draw_boxes ----------------------------------------------------------------


def test_draw_boxes_uses_one_colour_per_slot() -> None:
    canvas = np.zeros((100, 100, 3), np.uint8)
    hr = HandResult(
        1,
        1.0,
        present=True,
        hands=(Hand((10, 10, 50, 50), None, 0.9), Hand((60, 60, 90, 90), None, 0.9)),
    )
    draw_boxes(canvas, hr)
    assert (
        tuple(int(v) for v in canvas[10, 30]) == SLOT_COLORS[0]
    )  # on box 1's top edge
    assert (
        tuple(int(v) for v in canvas[60, 75]) == SLOT_COLORS[1]
    )  # on box 2's top edge
    assert not canvas[5, 5].any()  # outside both boxes: untouched


def test_draw_boxes_with_no_result_leaves_the_canvas_alone() -> None:
    canvas = np.zeros((20, 20, 3), np.uint8)
    draw_boxes(canvas, None)
    assert not canvas.any()


# --- PushFrameSource ---------------------------------------------------------------


def _read_in_thread(src: PushFrameSource) -> tuple[threading.Thread, list[object]]:
    out: list[object] = []
    t = threading.Thread(target=lambda: out.append(src.read()), daemon=True)
    t.start()
    return t, out


def test_push_source_read_blocks_until_a_frame_is_pushed() -> None:
    src = PushFrameSource()
    src.open()
    t, out = _read_in_thread(src)
    time.sleep(0.05)
    assert t.is_alive()  # nothing pushed: still blocked
    img = np.zeros((2, 2, 3), np.uint8)
    src.push(img)
    t.join(2.0)
    assert out == [img]
    assert out[0] is img  # handed over by reference, no copy


def test_push_source_close_unblocks_read_and_open_restarts_it() -> None:
    src = PushFrameSource()
    src.open()
    t, out = _read_in_thread(src)
    src.close()
    t.join(2.0)
    assert out == [None]  # closed: the blocked read returned None
    assert src.read() is None  # and stays closed
    src.open()
    img = np.ones((2, 2, 3), np.uint8)
    src.push(img)
    assert src.read() is img  # restartable


# --- Snapshots ----------------------------------------------------------------


class FakeProvider:
    def __init__(self, frame: Frame | None = None) -> None:
        self.frame = frame

    def latest(self) -> Frame | None:
        return self.frame


class FakeHandStage:
    def __init__(self, result: HandResult | None = None) -> None:
        self.result = result

    def latest(self) -> HandResult | None:
        return self.result


class ScriptedDetector:
    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]:
        return (DetectedHand((0.25, 0.25, 0.75, 0.75), 0.9),)

    def close(self) -> None:
        pass


class ScriptedClassifier:
    labels = ("fist", "palm")

    def classify(self, image_bgr: np.ndarray) -> dict[str, float]:
        return {"fist": 0.9, "palm": 0.1}

    def close(self) -> None:
        pass


def wait_until(pred: object, timeout: float = 2.0) -> None:
    assert callable(pred)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    assert pred(), "condition not met within timeout"


def flat_image(w: int = 64, h: int = 48) -> np.ndarray:
    # Flat colour so the lossy JPEG round-trip stays within rounding of the source.
    return np.full((h, w, 3), (30, 120, 200), np.uint8)


def assert_jpeg_matches(path: Path, source: np.ndarray) -> None:
    back = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert back is not None
    assert back.shape == source.shape
    assert np.abs(back.astype(int) - source.astype(int)).max() <= 2


def test_snapshot_path_names_by_stage_and_wall_clock_second() -> None:
    now = datetime(2026, 9, 17, 3, 7, 9, tzinfo=UTC)
    assert snapshot_path("/tmp/out", "HandStage", now) == Path(
        "/tmp/out/snapshot_HandStage_20260917030709.jpg"
    )


def test_default_snapshot_folder_is_the_demo_scripts_directory() -> None:
    assert DEMO_DIR == Path(hand_demo.__file__).resolve().parent
    assert (DEMO_DIR / "hand_demo.py").is_file()


def test_save_snapshot_of_the_detector_writes_the_frame_it_saw(tmp_path: Path) -> None:
    img = flat_image()
    hands = HandStage(
        FakeProvider(Frame(1, 1.0, img)), target_fps=None, detector=ScriptedDetector()
    )
    hands.start()
    try:
        wait_until(lambda: hands.latest() is not None)
        message = save_snapshot(
            hands, tmp_path, now=datetime(2026, 9, 17, 3, 7, 9, tzinfo=UTC)
        )
    finally:
        hands.stop()

    target = tmp_path / "snapshot_HandStage_20260917030709.jpg"
    assert message == f"Saved `{target}`"
    assert_jpeg_matches(target, img)


def test_save_snapshot_of_the_classifier_writes_the_crop_it_classified(
    tmp_path: Path,
) -> None:
    crop = flat_image(16, 16)
    hr = HandResult(1, 1.0, present=True, hands=(Hand((0, 0, 16, 16), crop, 0.9),))
    gestures = GestureClassifier(
        FakeHandStage(hr), target_fps=None, classifier=ScriptedClassifier()
    )
    gestures.start()
    try:
        wait_until(lambda: gestures.latest() is not None)
        message = save_snapshot(
            gestures, tmp_path, now=datetime(2026, 9, 17, 3, 7, 9, tzinfo=UTC)
        )
    finally:
        gestures.stop()

    target = tmp_path / "snapshot_GestureClassifier_20260917030709_0.jpg"
    assert message == f"Saved `{target}`"
    assert_jpeg_matches(target, crop)


def test_save_snapshot_reports_instead_of_raising_when_nothing_was_processed(
    tmp_path: Path,
) -> None:
    gestures = GestureClassifier(FakeHandStage(), classifier=ScriptedClassifier())
    message = save_snapshot(gestures, tmp_path)
    assert message.startswith("⚠️ GestureClassifier:")
    assert "not processed anything yet" in message
    assert list(tmp_path.iterdir()) == []


def test_save_snapshot_reports_when_the_classifiers_last_input_had_no_hand(
    tmp_path: Path,
) -> None:
    empty = HandResult(1, 1.0, present=False, hands=())
    gestures = GestureClassifier(
        FakeHandStage(empty), target_fps=None, classifier=ScriptedClassifier()
    )
    gestures.start()
    try:
        wait_until(lambda: gestures.latest() is not None)
        message = save_snapshot(gestures, tmp_path)
    finally:
        gestures.stop()
    assert "no crop to save" in message
    assert list(tmp_path.iterdir()) == []
