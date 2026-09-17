import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_modules import (
    DetectedHand,
    Frame,
    GestureClassifier,
    Hand,
    HandResult,
    HandStage,
    Result,
    Stage,
    StreamProvider,
    save_crops,
    save_frame,
    save_input,
)

# --- Test helpers ------------------------------------------------------------


def gradient(w: int = 40, h: int = 30) -> np.ndarray:
    """A BGR image whose three channels each vary differently, so a channel swap
    or a transposed shape would break pixel equality."""
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.arange(w)[None, :]
    img[..., 1] = np.arange(h)[:, None] * 3
    img[..., 2] = 200
    return img


def frame(frame_id: int = 1, img: np.ndarray | None = None) -> Frame:
    return Frame(frame_id, float(frame_id), gradient() if img is None else img)


def hand(
    crop: np.ndarray | None, bbox: tuple[int, int, int, int] = (0, 0, 8, 8)
) -> Hand:
    return Hand(bbox=bbox, crop=crop, score=0.9)


class OneFrameSource:
    """Yields `image` once, then blocks until closed (like a live camera that stalls)."""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self._sent = False
        self._closed = threading.Event()

    def open(self) -> None:
        self._sent = False
        self._closed.clear()

    def read(self) -> np.ndarray | None:
        if not self._sent:
            self._sent = True
            return self.image
        self._closed.wait()
        return None

    def close(self) -> None:
        self._closed.set()

    def fps(self) -> float | None:
        return None


class ScriptedDetector:
    def __init__(self, hands: tuple[DetectedHand, ...]) -> None:
        self.hands = hands

    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]:
        return self.hands

    def close(self) -> None:
        pass


class ScriptedClassifier:
    labels = ("fist", "palm")

    def classify(self, image_bgr: np.ndarray) -> dict[str, float]:
        return {"fist": 0.9, "palm": 0.1}

    def close(self) -> None:
        pass


class FakeHandStage:
    def __init__(self, result: HandResult | None = None) -> None:
        self.result = result

    def latest(self) -> HandResult | None:
        return self.result


def hand_result(frame_id: int, *crops: np.ndarray | None) -> HandResult:
    return HandResult(
        frame_id=frame_id,
        ts=float(frame_id),
        present=bool(crops),
        hands=tuple(hand(c) for c in crops),
    )


def wait_until(pred: object, timeout: float = 2.0) -> None:
    assert callable(pred)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    assert pred(), "condition not met within timeout"


# --- save_frame ----------------------------------------------------------------


def test_save_frame_png_round_trips_pixels_exactly(tmp_path: Path) -> None:
    img = gradient()
    before = img.copy()
    out = save_frame(frame(img=img), tmp_path / "shot.png")

    assert out == tmp_path / "shot.png"
    back = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert back is not None
    assert back.shape == img.shape
    assert np.array_equal(back, img)
    assert np.array_equal(img, before), "saving must not touch the shared frame buffer"


def test_save_frame_creates_missing_parent_directories(tmp_path: Path) -> None:
    out = save_frame(frame(), tmp_path / "a" / "b" / "shot.png")
    assert out.is_file()


def test_save_frame_jpeg_writes_a_readable_image_of_the_same_shape(
    tmp_path: Path,
) -> None:
    out = save_frame(frame(), tmp_path / "shot.jpg")
    back = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert back is not None
    assert back.shape == gradient().shape


def test_save_frame_accepts_str_paths(tmp_path: Path) -> None:
    out = save_frame(frame(), str(tmp_path / "shot.png"))
    assert isinstance(out, Path)
    assert out.is_file()


def test_save_frame_rejects_none() -> None:
    with pytest.raises(ValueError, match="no frame"):
        save_frame(None, "shot.png")


def test_save_frame_rejects_unsupported_suffix_before_touching_disk(
    tmp_path: Path,
) -> None:
    target = tmp_path / "out" / "shot.txt"
    with pytest.raises(ValueError, match="unsupported image format"):
        save_frame(frame(), target)
    assert not target.parent.exists()


# --- save_crops ----------------------------------------------------------------


def test_save_crops_writes_one_indexed_file_per_hand_aligned_with_hands(
    tmp_path: Path,
) -> None:
    crop0 = gradient(12, 10)
    crop2 = gradient(6, 5)
    result = HandResult(
        frame_id=7,
        ts=7.0,
        present=True,
        hands=(hand(crop0), hand(None, bbox=(40, 30, 40, 30)), hand(crop2)),
    )

    paths = save_crops(result, tmp_path / "crop.png")

    assert paths == (tmp_path / "crop_0.png", None, tmp_path / "crop_2.png")
    assert not (tmp_path / "crop_1.png").exists()
    assert not (tmp_path / "crop.png").exists()
    back0 = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    back2 = cv2.imread(str(paths[2]), cv2.IMREAD_COLOR)
    assert back0 is not None and np.array_equal(back0, crop0)
    assert back2 is not None and np.array_equal(back2, crop2)


def test_save_crops_on_empty_result_writes_nothing(tmp_path: Path) -> None:
    result = HandResult(frame_id=1, ts=1.0, present=False, hands=())
    assert save_crops(result, tmp_path / "crop.png") == ()
    assert list(tmp_path.iterdir()) == []


def test_save_crops_rejects_none() -> None:
    with pytest.raises(ValueError, match="no hand result"):
        save_crops(None, "crop.png")


def test_save_crops_rejects_unsupported_suffix(tmp_path: Path) -> None:
    result = HandResult(frame_id=1, ts=1.0, present=True, hands=(hand(gradient(4, 4)),))
    with pytest.raises(ValueError, match="unsupported image format"):
        save_crops(result, tmp_path / "crop.gif.bogus")
    assert list(tmp_path.iterdir()) == []


# --- End to end through the pipeline ------------------------------------------


def test_snapshots_from_latest_values_match_the_frame_and_its_bbox(
    tmp_path: Path,
) -> None:
    img = gradient(64, 48)
    provider = StreamProvider(OneFrameSource(img))
    detector = ScriptedDetector((DetectedHand((0.25, 0.25, 0.75, 0.75), 0.9),))
    hands = HandStage(provider, target_fps=None, pad=0.0, detector=detector)
    provider.start()
    hands.start()
    try:
        wait_until(lambda: hands.latest() is not None)
        f = provider.latest()
        hr = hands.latest()
        assert f is not None and hr is not None and hr.first is not None
        assert hr.frame_id == f.frame_id  # the crop belongs to this very frame

        frame_file = save_frame(f, tmp_path / f"{f.frame_id:06d}.png")
        (crop_file,) = save_crops(hr, tmp_path / f"{hr.frame_id:06d}_hand.png")
        input_file = save_input(hands, tmp_path / "hand_input.png")
    finally:
        hands.stop()
        provider.stop()

    assert frame_file == tmp_path / "000001.png"
    assert crop_file == tmp_path / "000001_hand_0.png"
    assert input_file == tmp_path / "hand_input.png"
    full = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
    crop = cv2.imread(str(crop_file), cv2.IMREAD_COLOR)
    hand_input = cv2.imread(str(input_file), cv2.IMREAD_COLOR)
    assert full is not None and crop is not None and hand_input is not None
    assert np.array_equal(hand_input, img)  # the hand stage's input IS the frame
    x0, y0, x1, y1 = hr.first.bbox
    assert (x0, y0, x1, y1) == (16, 12, 48, 36)
    assert np.array_equal(full, img)
    assert np.array_equal(crop, full[y0:y1, x0:x1])


# --- save_input --------------------------------------------------------------


def test_save_input_on_a_classifier_writes_the_crop_it_classified(
    tmp_path: Path,
) -> None:
    crop = gradient(12, 10)
    upstream = FakeHandStage(hand_result(1, crop))
    gestures = GestureClassifier(
        upstream, target_fps=None, classifier=ScriptedClassifier()
    )
    gestures.start()
    try:
        wait_until(lambda: gestures.latest() is not None)
        paths = save_input(gestures, tmp_path / "gesture_in.png")
    finally:
        gestures.stop()

    assert paths == (tmp_path / "gesture_in_0.png",)
    assert isinstance(paths, tuple)
    back = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    assert back is not None and np.array_equal(back, crop)


def test_save_input_keeps_the_classifiers_own_input_after_the_upstream_moved_on(
    tmp_path: Path,
) -> None:
    classified = gradient(12, 10)
    upstream = FakeHandStage(hand_result(1, classified))
    gestures = GestureClassifier(
        upstream, target_fps=None, classifier=ScriptedClassifier()
    )
    gestures.start()
    try:
        wait_until(lambda: gestures.latest() is not None)
    finally:
        gestures.stop()

    # The hand stage has since published a different crop the classifier never saw.
    upstream.result = hand_result(2, gradient(6, 5))
    newest = upstream.latest()
    assert newest is not None and newest.frame_id == 2

    paths = save_input(gestures, tmp_path / "in.png")
    assert isinstance(paths, tuple) and len(paths) == 1
    path = paths[0]
    assert path is not None
    back = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert back is not None and np.array_equal(back, classified)
    g = gestures.latest()
    assert g is not None and gestures.last_input is not None
    assert g.frame_id == gestures.last_input.frame_id == 1  # output and input match


def test_save_input_before_any_process_call_raises(tmp_path: Path) -> None:
    gestures = GestureClassifier(FakeHandStage(), classifier=ScriptedClassifier())
    with pytest.raises(ValueError, match="has not processed anything yet"):
        save_input(gestures, tmp_path / "in.png")
    assert list(tmp_path.iterdir()) == []


def test_save_input_rejects_an_input_type_without_pixels(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Tick:
        frame_id: int

    class TickSource:
        def latest(self) -> Tick:
            return Tick(1)

    class TickStage(Stage[Tick, Result]):
        def process(self, item: Tick) -> Result:
            return Result(item.frame_id, 0.0, True)

    st = TickStage(TickSource(), target_fps=None, name="ticks")
    st.start()
    try:
        wait_until(lambda: st.last_input is not None)
        with pytest.raises(TypeError, match="ticks.*Tick"):
            save_input(st, tmp_path / "in.png")
    finally:
        st.stop()
    assert list(tmp_path.iterdir()) == []
