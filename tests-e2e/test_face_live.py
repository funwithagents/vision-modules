"""MediaPipeFaceDetector on the real model (specs/face.md "Tests").

The orientation signs are only checked against the real model here: the fixtures
are public-domain photos whose pose was judged by eye (fixtures/README.md).
"""

import math
import threading

import cv2
import numpy as np
import pytest
from support import face_image_path

from vision_modules.face import DetectedFace, MediaPipeFaceDetector

TOL = math.radians(5)
CLEAR = math.radians(10)


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(face_image_path(name)))
    assert img is not None
    return img


def detect_once(img: np.ndarray, max_faces: int = 1) -> tuple[DetectedFace, ...]:
    """A fresh detector per independent image: the landmarker tracks between calls."""
    det = MediaPipeFaceDetector(max_faces=max_faces)
    try:
        return det.detect(img, 0.0)
    finally:
        det.close()


def single_face(img: np.ndarray) -> DetectedFace:
    faces = detect_once(img)
    assert len(faces) == 1
    assert faces[0].orientation is not None and faces[0].eyes is not None
    return faces[0]


def eye_line_roll(face: DetectedFace, img: np.ndarray) -> float:
    """The spec's roll from the eyes, in pixels so a non-square image doesn't skew it."""
    assert face.eyes is not None
    (rx, ry), (lx, ly) = face.eyes
    h, w = img.shape[:2]
    return math.atan2((ly - ry) * h, (lx - rx) * w)


def rotate_clockwise(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(
        (w / 2, h / 2), -deg, 1.0
    )  # OpenCV: + is counter-clockwise
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def test_blank_image_has_no_face() -> None:
    assert detect_once(np.zeros((480, 640, 3), np.uint8)) == ()


def test_frontal_face_geometry_and_orientation() -> None:
    img = load("face_frontal")
    face = single_face(img)
    assert face.orientation is not None and face.eyes is not None and face.nose

    x0, y0, x1, y1 = face.box
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    for x, y in (face.nose, *face.eyes):
        assert x0 <= x <= x1 and y0 <= y <= y1
    right, left = face.eyes
    assert right[0] < left[0]  # the subject's right eye is on the image's left
    assert face.landmarks is not None and face.landmarks.shape == (478, 3)
    assert face.score is None

    roll, pitch, yaw = face.orientation
    assert abs(pitch) < CLEAR and abs(yaw) < CLEAR
    assert roll == pytest.approx(eye_line_roll(face, img), abs=TOL)


@pytest.mark.parametrize("deg", [15.0, -15.0])
def test_rotated_frontal_face_rolls_by_the_rotation(deg: float) -> None:
    img = load("face_frontal")
    base = single_face(img)
    assert base.orientation is not None
    turned_img = rotate_clockwise(img, deg)
    turned = single_face(turned_img)
    assert turned.orientation is not None

    expected = base.orientation[0] + math.radians(deg)
    assert turned.orientation[0] == pytest.approx(expected, abs=TOL)
    assert eye_line_roll(turned, turned_img) == pytest.approx(expected, abs=TOL)


def test_two_side_by_side_faces_are_both_found() -> None:
    img = load("face_frontal")
    faces = detect_once(np.hstack([img, img]), max_faces=2)
    assert len(faces) == 2
    a, b = sorted(faces, key=lambda f: f.box[0])
    assert a.box[2] < b.box[0]  # disjoint: one per half


def test_face_turned_toward_image_right_has_positive_yaw() -> None:
    img = load("face_yaw_right")
    face = single_face(img)
    assert face.orientation is not None
    assert face.orientation[2] > CLEAR

    mirrored = single_face(np.ascontiguousarray(img[:, ::-1]))
    assert mirrored.orientation is not None
    assert mirrored.orientation[2] < -CLEAR


def test_face_tilted_down_has_positive_pitch() -> None:
    face = single_face(load("face_pitch_down"))
    assert face.orientation is not None
    assert face.orientation[1] > CLEAR


def test_sequential_calls_from_different_threads() -> None:
    img = load("face_frontal")
    det = MediaPipeFaceDetector()
    found: list[int] = []
    errors: list[Exception] = []
    builder = threading.current_thread()
    off_builder: list[bool] = []

    def call(ts: float) -> None:
        off_builder.append(threading.current_thread() is not builder)
        try:
            found.append(len(det.detect(img, ts)))
        except Exception as e:  # noqa: BLE001 — surfaced below, not lost in the thread
            errors.append(e)

    try:
        for i in range(10):
            t = threading.Thread(target=call, args=(i / 30,))
            t.start()
            t.join()  # one call at a time, never concurrent
    finally:
        det.close()

    assert not errors, errors
    assert found == [1] * 10
    assert off_builder == [True] * 10  # each call on its own new thread
