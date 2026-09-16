import cv2
import numpy as np
from support import require_env

from vision_modules.hand import MediaPipeHandDetector


def test_mediapipe_detector_runs_on_a_blank_image() -> None:
    require_env("VISION_MODULES_E2E")  # skip unless opted in (network: model download)
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
    require_env("VISION_MODULES_E2E")
    path = require_env(
        "VISION_MODULES_HAND_IMAGE"
    )  # path to any photo with one clearly visible hand
    img = cv2.imread(path)
    assert img is not None
    det = MediaPipeHandDetector()
    try:
        out = det.detect(img, 0.0)
        assert len(out) >= 1
        x0, y0, x1, y1 = out[0].box
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    finally:
        det.close()
