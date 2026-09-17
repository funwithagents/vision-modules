import cv2
import numpy as np
from support import hand_image_path, two_hands_image_path

from vision_modules.hand import MediaPipeHandDetector, box_iou


def test_mediapipe_detector_runs_on_a_blank_image() -> None:
    det = MediaPipeHandDetector()
    try:
        blank = np.zeros((480, 640, 3), np.uint8)
        out = det.detect(blank, 0.0)
        assert out == ()  # no hand in a black image
        out2 = det.detect(blank, 0.0)  # same ts twice must not raise (timestamp bump)
        assert out2 == ()
    finally:
        det.close()


def test_mediapipe_detector_finds_a_hand_in_a_real_photo() -> None:
    path = hand_image_path()
    img = cv2.imread(str(path))
    assert img is not None
    det = MediaPipeHandDetector()
    try:
        out = det.detect(img, 0.0)
        assert len(out) >= 1
        x0, y0, x1, y1 = out[0].box
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    finally:
        det.close()


def test_mediapipe_detector_returns_at_most_num_hands() -> None:
    img = cv2.imread(str(two_hands_image_path()))
    assert img is not None
    two = MediaPipeHandDetector(num_hands=2)
    one = MediaPipeHandDetector(num_hands=1)
    try:
        found_two = two.detect(img, 0.0)
        found_one = one.detect(img, 0.0)
    finally:
        two.close()
        one.close()
    assert len(found_two) == 2
    assert box_iou(found_two[0].box, found_two[1].box) < 0.5  # two distinct hands
    assert len(found_one) == 1  # never more than num_hands: the spec's one-count trade
