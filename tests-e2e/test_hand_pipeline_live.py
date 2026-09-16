"""Full detect + crop + classify chain, the same wiring examples/hand_demo.py uses:
StreamProvider -> HandStage -> GestureClassifier, real MediaPipe + real HaGRID model.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from support import hand_image_path, labeled_hand_fixtures, no_hand_image_path

from vision_modules import (
    Gesture,
    GestureClassifier,
    HandResult,
    HandStage,
    Pipeline,
    StreamProvider,
)


class StaticImageSource:
    """FrameSource replaying one fixed image, paced like a real camera."""

    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def open(self) -> None:
        pass

    def read(self) -> np.ndarray | None:
        time.sleep(1 / 30)
        return self._image

    def close(self) -> None:
        pass


def _run_pipeline(img: np.ndarray) -> tuple[HandResult, Gesture]:
    """Runs the real detect+crop+classify chain until both stages publish once."""
    provider = StreamProvider(StaticImageSource(img))
    hands = HandStage(provider, target_fps=10)
    gestures = GestureClassifier(hands, target_fps=5)

    with Pipeline([provider, hands, gestures]):
        deadline = time.monotonic() + 60.0  # first run downloads both models
        while time.monotonic() < deadline and gestures.latest() is None:
            if hands.last_error is not None:
                raise hands.last_error
            if gestures.last_error is not None:
                raise gestures.last_error
            time.sleep(0.1)

        hr = hands.latest()
        gesture = gestures.latest()
        assert hr is not None, "hand stage published nothing within 60s"
        assert gesture is not None, "classifier published nothing within 60s"
        return hr, gesture


def test_full_pipeline_detects_and_crops_a_real_hand() -> None:
    img = cv2.imread(str(hand_image_path()))
    assert img is not None
    hr, gesture = _run_pipeline(img)
    assert hr.present and hr.first is not None and hr.first.crop is not None
    assert gesture.present and gesture.first is not None and gesture.first.scores
    assert abs(sum(gesture.first.scores.values()) - 1.0) < 1e-2


def test_full_pipeline_reports_no_hand_present() -> None:
    img = cv2.imread(str(no_hand_image_path()))
    assert img is not None
    hr, gesture = _run_pipeline(img)
    assert not hr.present and hr.first is None
    assert not gesture.present and gesture.first is None


_FIXTURES: list[tuple[str, Path | None]] = list(labeled_hand_fixtures()) or [
    ("<none>", None)
]


@pytest.mark.parametrize(
    ("label", "path"), _FIXTURES, ids=[label for label, _ in _FIXTURES]
)
def test_full_pipeline_predicts_the_photographed_gesture(
    label: str, path: Path | None
) -> None:
    if path is None:
        pytest.skip(
            "no labeled hand fixtures in tests-e2e/fixtures/ (hand_<gesture>.jpg)"
        )
    img = cv2.imread(str(path))
    assert img is not None
    _, gesture = _run_pipeline(img)
    assert gesture.first is not None
    top_label, _ = max(gesture.first.scores.items(), key=lambda kv: kv[1])
    assert top_label == label
