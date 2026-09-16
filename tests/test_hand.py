import time

import numpy as np

from vision_modules.hand import DetectedHand, HandStage, padded_box
from vision_modules.stream import Frame

# --- Test helpers ------------------------------------------------------------


class ScriptedDetector:
    def __init__(self, hands: tuple[DetectedHand, ...]) -> None:
        self.hands = hands
        self.calls: list[float] = []
        self.closed = False

    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]:
        self.calls.append(ts)
        return self.hands

    def close(self) -> None:
        self.closed = True


class FakeProvider:
    def __init__(self) -> None:
        self.frame: Frame | None = None

    def latest(self) -> Frame | None:
        return self.frame


def make_frame(frame_id: int, w: int = 64, h: int = 48) -> Frame:
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.arange(w)
    return Frame(frame_id, float(frame_id), img)


def wait_until(pred: object, timeout: float = 2.0) -> None:
    assert callable(pred)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    assert pred(), "condition not met within timeout"


# --- padded_box ----------------------------------------------------------------


def test_padded_box_scales_pads_and_rounds() -> None:
    assert padded_box((0.25, 0.25, 0.75, 0.75), 100, 100, 0.0) == (25, 25, 75, 75)
    assert padded_box((0.25, 0.25, 0.75, 0.75), 100, 100, 0.5) == (0, 0, 100, 100)
    assert padded_box((0.25, 0.25, 0.75, 0.75), 100, 100, 0.1) == (20, 20, 80, 80)


def test_padded_box_clamps_to_frame() -> None:
    assert padded_box((0.9, 0.9, 1.0, 1.0), 100, 100, 0.5) == (85, 85, 100, 100)


def test_padded_box_degenerate_box_has_zero_area() -> None:
    x0, y0, x1, y1 = padded_box((1.0, 1.0, 1.0, 1.0), 100, 100, 0.5)
    assert x1 <= x0
    assert y1 <= y0


# --- HandStage -------------------------------------------------------------------


def test_stage_publishes_a_copied_crop_matching_the_bbox() -> None:
    detector = ScriptedDetector((DetectedHand((0.25, 0.25, 0.75, 0.75), 0.9),))
    provider = FakeProvider()
    frame = make_frame(1, w=64, h=48)
    provider.frame = frame
    stage = HandStage(provider, target_fps=None, pad=0.0, detector=detector)
    stage.start()
    try:
        wait_until(lambda: stage.latest() is not None)
        time.sleep(0.02)
        r = stage.latest()
        assert r is not None
        assert r.present
        assert len(r.hands) == 1
        h = r.first
        assert h is not None
        assert h.bbox == (16, 12, 48, 36)
        assert h.crop is not None
        assert h.crop.shape == (24, 32, 3)
        assert np.array_equal(h.crop, frame.image[12:36, 16:48])
        assert not np.shares_memory(h.crop, frame.image)
        assert h.score == 0.9
    finally:
        stage.stop()


def test_stage_reports_empty_frame_as_present_false() -> None:
    detector = ScriptedDetector(())
    provider = FakeProvider()
    frame = make_frame(1)
    provider.frame = frame
    stage = HandStage(provider, target_fps=None, detector=detector)
    stage.start()
    try:
        wait_until(lambda: stage.latest() is not None)
        r = stage.latest()
        assert r is not None
        assert r.present is False
        assert r.hands == ()
        assert r.first is None
        assert r.frame_id == frame.frame_id
    finally:
        stage.stop()


def test_stage_orders_by_score_and_truncates_to_max_hands() -> None:
    detector = ScriptedDetector(
        (
            DetectedHand((0.0, 0.0, 0.5, 0.5), 0.2),
            DetectedHand((0.0, 0.0, 0.5, 0.5), 0.9),
            DetectedHand((0.0, 0.0, 0.5, 0.5), 0.5),
        )
    )
    provider = FakeProvider()
    provider.frame = make_frame(1)
    stage = HandStage(provider, target_fps=None, max_hands=2, detector=detector)
    stage.start()
    try:
        wait_until(lambda: stage.latest() is not None)
        r = stage.latest()
        assert r is not None
        assert [h.score for h in r.hands] == [0.9, 0.5]
    finally:
        stage.stop()


def test_stage_passes_frame_timestamp_to_detector() -> None:
    detector = ScriptedDetector(())
    provider = FakeProvider()
    frame = make_frame(1)
    provider.frame = frame
    stage = HandStage(provider, target_fps=None, detector=detector)
    stage.start()
    try:
        wait_until(lambda: stage.latest() is not None)
        assert detector.calls == [frame.ts]
    finally:
        stage.stop()


def test_stage_gives_none_crop_for_zero_area_box() -> None:
    detector = ScriptedDetector((DetectedHand((1.0, 1.0, 1.0, 1.0), 0.9),))
    provider = FakeProvider()
    provider.frame = make_frame(1)
    stage = HandStage(provider, target_fps=None, detector=detector)
    stage.start()
    try:
        wait_until(lambda: stage.latest() is not None)
        r = stage.latest()
        assert r is not None
        h = r.first
        assert h is not None
        assert h.crop is None
        assert isinstance(h.bbox, tuple)
        assert len(h.bbox) == 4
    finally:
        stage.stop()


def test_stage_closes_detector_on_stop() -> None:
    detector = ScriptedDetector(())
    provider = FakeProvider()
    provider.frame = make_frame(1)
    stage = HandStage(provider, target_fps=None, detector=detector)
    stage.start()
    try:
        wait_until(lambda: stage.latest() is not None)
    finally:
        stage.stop()
    assert detector.closed is True


def test_stage_processes_each_frame_once() -> None:
    detector = ScriptedDetector(())
    provider = FakeProvider()
    provider.frame = make_frame(1)
    stage = HandStage(provider, target_fps=None, detector=detector)
    stage.start()
    try:
        wait_until(lambda: stage.latest() is not None)
        time.sleep(0.05)
        assert len(detector.calls) == 1
    finally:
        stage.stop()
