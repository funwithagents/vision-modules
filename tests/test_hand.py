import threading
import time

import numpy as np
import pytest

import vision_modules.hand as hand_module
from vision_modules.hand import (
    MATCH_IOU,
    DetectedHand,
    HandStage,
    box_area,
    box_iou,
    padded_box,
    select_hands,
)
from vision_modules.stream import Frame

# --- Test helpers ------------------------------------------------------------


class ScriptedDetector:
    """Returns a fixed tuple, or — given a list — one tuple per detect() call,
    repeating the last entry once the list is exhausted."""

    def __init__(
        self,
        hands: tuple[DetectedHand, ...] | list[tuple[DetectedHand, ...]],
    ) -> None:
        self.script = hands if isinstance(hands, list) else [hands]
        self.calls: list[float] = []
        self.closed = False

    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]:
        index = min(len(self.calls), len(self.script) - 1)
        self.calls.append(ts)
        return self.script[index]

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


A = DetectedHand((0.0, 0.0, 0.4, 0.4), 0.5)  # area 0.16
B = DetectedHand((0.6, 0.0, 0.9, 0.3), 0.9)  # area 0.09, top right
C = DetectedHand((0.0, 0.5, 0.5, 1.0), 0.1)  # area 0.25, bottom left
A_MOVED = DetectedHand(
    (0.02, 0.0, 0.42, 0.4), 0.5
)  # A shifted a little: area 0.16, IoU with A ≈ 0.9
B_GROWN = DetectedHand(
    (0.55, 0.0, 1.0, 0.45), 0.9
)  # B grown: area 0.2025, IoU with B ≈ 0.44
N_10 = DetectedHand(
    (0.55, 0.55, 0.97, 0.97), 0.5
)  # newcomer, area 0.1764 (10 % bigger than A)
N_30 = DetectedHand(
    (0.5, 0.5, 0.96, 0.96), 0.5
)  # newcomer, area 0.2116 (32 % bigger than A)
A_FAR = DetectedHand((0.5, 0.5, 0.9, 0.9), 0.5)  # same size as A, no overlap with A


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


# --- select_hands ----------------------------------------------------------------


def test_box_area_and_iou() -> None:
    assert box_area(A.box) == pytest.approx(0.16)
    assert box_area((0.5, 0.5, 0.4, 0.4)) == 0.0  # inverted box
    assert box_iou(A.box, A.box) == pytest.approx(1.0)
    assert box_iou(A.box, C.box) == 0.0  # disjoint
    assert box_iou(B.box, B_GROWN.box) == pytest.approx(0.09 / 0.2025)
    assert (
        box_iou(B.box, B_GROWN.box) >= MATCH_IOU
    )  # the slot-stability tests rely on this


def test_select_no_previous_ranks_by_area_and_truncates() -> None:
    assert select_hands((B, A, C), (), 3, "area", 0.2) == (C, A, B)
    assert select_hands((B, A, C), (), 2, "area", 0.2) == (C, A)
    assert select_hands((B, A, C), (), 1, "area", 0.2) == (C,)


def test_select_rank_by_score() -> None:
    assert select_hands((B, A, C), (), 3, "score", 0.2) == (B, A, C)
    assert select_hands((A, C, B), (), 1, "score", 0.2) == (B,)


def test_select_ties_keep_detector_order() -> None:
    d1 = DetectedHand((0.0, 0.0, 0.25, 0.25), 0.5)
    d2 = DetectedHand(
        (0.5, 0.5, 0.75, 0.75), 0.5
    )  # same area (exact in binary), same score
    assert select_hands((d1, d2), (), 2, "area", 0.2) == (d1, d2)
    assert select_hands((d2, d1), (), 2, "area", 0.2) == (d2, d1)


def test_select_hysteresis_keeps_the_incumbent_unless_clearly_beaten() -> None:
    previous = (A,)
    # 10 % bigger newcomer: kept out at 0.2, wins at 0
    assert select_hands((A_MOVED, N_10), previous, 1, "area", 0.2) == (A_MOVED,)
    assert select_hands((A_MOVED, N_10), previous, 1, "area", 0.0) == (N_10,)
    # 30 % bigger newcomer beats the 20 % boost
    assert select_hands((A_MOVED, N_30), previous, 1, "area", 0.2) == (N_30,)


def test_select_far_moved_hand_is_a_newcomer_not_an_incumbent() -> None:
    # A_FAR has no overlap with A, so it gets no boost and the bigger N_10 wins
    assert select_hands((A_FAR, N_10), (A,), 1, "area", 0.2) == (N_10,)


def test_select_slot_order_is_stable_when_sizes_cross() -> None:
    previous = (A, B)  # A in slot 0, B in slot 1
    # detector now returns them the other way round, and B has grown past A
    assert select_hands((B_GROWN, A_MOVED), previous, 2, "area", 0.2) == (
        A_MOVED,
        B_GROWN,
    )


def test_select_newcomer_fills_a_vacated_slot_after_the_incumbents() -> None:
    previous = (A_MOVED, B_GROWN)
    # B left; C appears, bigger than A. A keeps slot 0, C is appended.
    assert select_hands((C, A_MOVED), previous, 2, "area", 0.2) == (A_MOVED, C)


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
    stage = HandStage(
        provider, target_fps=None, max_hands=2, rank="score", detector=detector
    )
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


def test_stage_never_closes_a_borrowed_detector_and_reuses_it_on_restart() -> None:
    detector = ScriptedDetector(())
    provider = FakeProvider()
    provider.frame = make_frame(1)
    stage = HandStage(provider, target_fps=None, detector=detector)
    stage.start()
    wait_until(lambda: stage.latest() is not None)
    stage.stop()
    assert detector.closed is False  # the caller owns it

    provider.frame = make_frame(2)
    stage.start()
    try:
        wait_until(lambda: stage.published_count == 2)
        r = stage.latest()
        assert r is not None and r.frame_id == 2
        assert detector.calls == [1.0, 2.0]  # same instance served both runs
    finally:
        stage.stop()
    assert detector.closed is False


def test_stage_owns_the_default_detector_one_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMediaPipe(ScriptedDetector):
        def __init__(self, num_hands: int) -> None:
            super().__init__(())
            self.num_hands = num_hands
            self.created_on = threading.get_ident()
            created.append(self)  # resolved at call time, defined just below

    created: list[FakeMediaPipe] = []

    monkeypatch.setattr(hand_module, "MediaPipeHandDetector", FakeMediaPipe)
    provider = FakeProvider()
    provider.frame = make_frame(1)
    stage = HandStage(provider, target_fps=None, max_hands=2)

    stage.start()
    wait_until(lambda: stage.latest() is not None)
    stage.stop()
    assert len(created) == 1
    assert created[0].num_hands == 2
    assert created[0].created_on != threading.get_ident()  # built on the worker
    assert created[0].closed is True  # released with the run

    stage.start()
    try:
        wait_until(lambda: stage.published_count == 2)
    finally:
        stage.stop()
    assert len(created) == 2 and created[1] is not created[0]
    assert created[1].closed is True


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


def test_stage_keeps_slot_order_across_frames() -> None:
    detector = ScriptedDetector([(A, B), (B_GROWN, A_MOVED)])
    provider = FakeProvider()
    provider.frame = make_frame(1, w=100, h=100)
    stage = HandStage(
        provider, target_fps=None, pad=0.0, max_hands=2, detector=detector
    )
    stage.start()
    try:
        wait_until(lambda: stage.published_count == 1)
        provider.frame = make_frame(2, w=100, h=100)
        wait_until(lambda: stage.published_count == 2)
        r = stage.latest()
        assert r is not None and r.frame_id == 2
        assert [h.bbox for h in r.hands] == [
            (2, 0, 42, 40),
            (55, 0, 100, 45),
        ]  # A first, then B
    finally:
        stage.stop()


def test_stage_forgets_incumbents_between_runs() -> None:
    # run 1 publishes A; run 2 sees A and a 10 % bigger newcomer with max_hands=1.
    # If the memory leaked across runs, A would be kept; after close() it must not be.
    detector = ScriptedDetector([(A,), (A_MOVED, N_10)])
    provider = FakeProvider()
    provider.frame = make_frame(1, w=100, h=100)
    stage = HandStage(
        provider, target_fps=None, pad=0.0, max_hands=1, detector=detector
    )
    stage.start()
    wait_until(lambda: stage.published_count == 1)
    stage.stop()

    provider.frame = make_frame(2, w=100, h=100)
    stage.start()
    try:
        wait_until(lambda: stage.published_count == 2)
        r = stage.latest()
        assert r is not None and r.first is not None
        assert r.first.bbox == (55, 55, 97, 97)  # N_10, not A_MOVED
    finally:
        stage.stop()


def test_stage_rejects_bad_arguments_before_starting() -> None:
    provider = FakeProvider()
    detector = ScriptedDetector(())
    with pytest.raises(ValueError):
        HandStage(provider, max_hands=0, detector=detector)
    with pytest.raises(ValueError):
        HandStage(provider, hysteresis=-0.1, detector=detector)
    with pytest.raises(ValueError):
        HandStage(provider, rank="size", detector=detector)  # pyright: ignore[reportArgumentType]
