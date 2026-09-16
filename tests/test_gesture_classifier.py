import threading
import time

import numpy as np
import pytest

import vision_modules.gesture_classifier as gc_module
from vision_modules.gesture_classifier import (
    GestureClassifier,
    HandGesture,
    select_device,
)
from vision_modules.hand import Hand, HandResult

# --- Test helpers ------------------------------------------------------------


class ScriptedClassifier:
    labels = ("fist", "palm", "stop")

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[np.ndarray] = []
        self.closed = False

    def classify(self, image_bgr: np.ndarray) -> dict[str, float]:
        self.seen.append(image_bgr)
        return self.scores

    def close(self) -> None:
        self.closed = True


class FakeHandStage:
    def __init__(self) -> None:
        self.result: HandResult | None = None

    def latest(self) -> HandResult | None:
        return self.result


def make_hand_result(frame_id: int, crops: list[np.ndarray | None]) -> HandResult:
    hands = tuple(Hand((0, 0, 8, 8), crop, 0.9) for crop in crops)
    return HandResult(frame_id, float(frame_id), present=bool(hands), hands=hands)


def wait_until(pred: object, timeout: float = 2.0) -> None:
    assert callable(pred)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    assert pred(), "condition not met within timeout"


# --- GestureClassifier -------------------------------------------------------


def test_labels_the_top_class_above_threshold() -> None:
    classifier = ScriptedClassifier({"fist": 0.1, "palm": 0.8, "stop": 0.1})
    stage = FakeHandStage()
    crop = np.zeros((4, 4, 3), np.uint8)
    stage.result = make_hand_result(1, [crop])
    gc = GestureClassifier(
        stage, target_fps=None, threshold=0.55, classifier=classifier
    )
    gc.start()
    try:
        wait_until(lambda: gc.latest() is not None)
        r = gc.latest()
        assert r is not None
        assert r.first is not None
        assert r.first.label == "palm"
        assert r.first.confidence == 0.8
        assert r.first.scores == {"fist": 0.1, "palm": 0.8, "stop": 0.1}
    finally:
        gc.stop()


def test_returns_none_label_below_threshold_but_keeps_scores() -> None:
    classifier = ScriptedClassifier({"fist": 0.4, "palm": 0.35, "stop": 0.25})
    stage = FakeHandStage()
    crop = np.zeros((4, 4, 3), np.uint8)
    stage.result = make_hand_result(1, [crop])
    gc = GestureClassifier(
        stage, target_fps=None, threshold=0.55, classifier=classifier
    )
    gc.start()
    try:
        wait_until(lambda: gc.latest() is not None)
        r = gc.latest()
        assert r is not None
        assert r.first is not None
        assert r.first.label is None
        assert r.first.confidence == 0.4
        assert r.first.scores == {"fist": 0.4, "palm": 0.35, "stop": 0.25}
    finally:
        gc.stop()


def test_score_equal_to_threshold_clears_it() -> None:
    classifier = ScriptedClassifier({"fist": 0.1, "palm": 0.55, "stop": 0.1})
    stage = FakeHandStage()
    crop = np.zeros((4, 4, 3), np.uint8)
    stage.result = make_hand_result(1, [crop])
    gc = GestureClassifier(
        stage, target_fps=None, threshold=0.55, classifier=classifier
    )
    gc.start()
    try:
        wait_until(lambda: gc.latest() is not None)
        r = gc.latest()
        assert r is not None
        assert r.first is not None
        assert r.first.label == "palm"
    finally:
        gc.stop()


def test_hand_without_crop_gets_empty_gesture() -> None:
    classifier = ScriptedClassifier({"fist": 1.0})
    stage = FakeHandStage()
    stage.result = make_hand_result(1, [None])
    gc = GestureClassifier(
        stage, target_fps=None, threshold=0.55, classifier=classifier
    )
    gc.start()
    try:
        wait_until(lambda: gc.latest() is not None)
        r = gc.latest()
        assert r is not None
        assert r.hands == (HandGesture(None, 0.0, {}),)
        assert classifier.seen == []
    finally:
        gc.stop()


def test_empty_hand_result_gives_present_false() -> None:
    classifier = ScriptedClassifier({"fist": 1.0})
    stage = FakeHandStage()
    stage.result = make_hand_result(7, [])
    gc = GestureClassifier(
        stage, target_fps=None, threshold=0.55, classifier=classifier
    )
    gc.start()
    try:
        wait_until(lambda: gc.latest() is not None)
        r = gc.latest()
        assert r is not None
        assert r.present is False
        assert r.hands == ()
        assert r.first is None
        assert r.frame_id == 7
        assert r.ts == 7.0
    finally:
        gc.stop()


def test_multi_hand_output_is_index_aligned() -> None:
    class PerCropClassifier:
        labels = ("fist", "palm")

        def __init__(self) -> None:
            self.seen: list[np.ndarray] = []
            self.closed = False

        def classify(self, image_bgr: np.ndarray) -> dict[str, float]:
            self.seen.append(image_bgr)
            if image_bgr[0, 0, 0] == 1:
                return {"palm": 1.0}
            return {"fist": 1.0}

        def close(self) -> None:
            self.closed = True

    classifier = PerCropClassifier()
    a = np.zeros((4, 4, 3), np.uint8)
    a[0, 0, 0] = 1
    b = np.zeros((4, 4, 3), np.uint8)
    stage = FakeHandStage()
    stage.result = make_hand_result(1, [a, b])
    gc = GestureClassifier(
        stage, target_fps=None, threshold=0.55, classifier=classifier
    )
    gc.start()
    try:
        wait_until(lambda: gc.latest() is not None)
        r = gc.latest()
        assert r is not None
        assert r.hands[0].label == "palm"
        assert r.hands[1].label == "fist"
    finally:
        gc.stop()


def test_classifier_receives_the_crop_unchanged() -> None:
    classifier = ScriptedClassifier({"palm": 1.0})
    crop = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    stage = FakeHandStage()
    stage.result = make_hand_result(1, [crop])
    gc = GestureClassifier(
        stage, target_fps=None, threshold=0.55, classifier=classifier
    )
    gc.start()
    try:
        wait_until(lambda: gc.latest() is not None)
        assert np.array_equal(classifier.seen[0], crop)
    finally:
        gc.stop()


def test_never_closes_a_borrowed_classifier_and_reuses_it_on_restart() -> None:
    classifier = ScriptedClassifier({"palm": 1.0})
    crop = np.zeros((4, 4, 3), np.uint8)
    stage = FakeHandStage()
    stage.result = make_hand_result(1, [crop])
    gc = GestureClassifier(
        stage, target_fps=None, threshold=0.55, classifier=classifier
    )
    gc.start()
    wait_until(lambda: gc.latest() is not None)
    gc.stop()
    assert classifier.closed is False  # the caller owns it

    stage.result = make_hand_result(2, [crop])
    gc.start()
    try:
        wait_until(lambda: gc.published_count == 2)
        r = gc.latest()
        assert r is not None and r.frame_id == 2
        assert len(classifier.seen) == 2  # same instance served both runs
    finally:
        gc.stop()
    assert classifier.closed is False


def test_owns_the_default_classifier_one_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHaGRID(ScriptedClassifier):
        def __init__(self, device: str) -> None:
            super().__init__({"palm": 1.0})
            self.device = device
            self.created_on = threading.get_ident()
            created.append(self)  # resolved at call time, defined just below

    created: list[FakeHaGRID] = []

    monkeypatch.setattr(gc_module, "HaGRIDViTClassifier", FakeHaGRID)
    stage = FakeHandStage()
    stage.result = make_hand_result(1, [np.zeros((4, 4, 3), np.uint8)])
    gc = GestureClassifier(stage, target_fps=None, device="cpu")

    gc.start()
    wait_until(lambda: gc.latest() is not None)
    gc.stop()
    assert len(created) == 1
    assert created[0].device == "cpu"  # explicit device passed straight through
    assert created[0].created_on != threading.get_ident()  # loaded on the worker
    assert created[0].closed is True  # released with the run

    gc.start()
    try:
        wait_until(lambda: gc.published_count == 2)
    finally:
        gc.stop()
    assert len(created) == 2 and created[1] is not created[0]
    assert created[1].closed is True


def test_select_device_honours_preference() -> None:
    assert select_device("cpu") == "cpu"
    assert select_device() in {"mps", "cuda", "cpu"}
