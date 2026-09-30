import math

import numpy as np
import pytest

import vision_modules.face as face_module
from vision_modules.face import (
    LEFT_IRIS,
    NOSE_TIP,
    RIGHT_IRIS,
    MediaPipeFaceDetector,
    _NativeFace,
    orientation_from_matrix,
)

# --- Test helpers ------------------------------------------------------------


class FakeLandmarker:
    """Stands in for MediaPipe: records each call, returns scripted native faces."""

    def __init__(self, faces: list[_NativeFace]) -> None:
        self.faces = faces
        self.calls: list[tuple[np.ndarray, int]] = []
        self.close_count = 0

    def detect(self, rgb: np.ndarray, ts_ms: int) -> list[_NativeFace]:
        self.calls.append((rgb, ts_ms))
        return self.faces

    def close(self) -> None:
        self.close_count += 1


def install_fake(
    monkeypatch: pytest.MonkeyPatch, faces: list[_NativeFace] | None = None
) -> FakeLandmarker:
    fake = FakeLandmarker(faces or [])
    monkeypatch.setattr(face_module, "_open_landmarker", lambda *args: fake)
    return fake


def mesh(n: int = 478) -> np.ndarray:
    """A mesh inside (0.3..0.6, 0.2..0.7), with known extremes and named points."""
    pts = np.full((n, 3), 0.5, dtype=np.float32)
    pts[10] = (0.3, 0.2, 0.0)
    pts[20] = (0.6, 0.7, 0.0)
    if n > NOSE_TIP:
        pts[NOSE_TIP] = (0.45, 0.5, -0.1)
    if n > LEFT_IRIS:
        pts[RIGHT_IRIS] = (0.38, 0.35, 0.0)
        pts[LEFT_IRIS] = (0.52, 0.36, 0.0)
    return pts


def rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose(r: np.ndarray, scale: float = 1.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r * scale
    m[:3, 3] = (1.0, -2.0, -40.0)  # a translation must not leak into the angles
    return m


def image() -> np.ndarray:
    img = np.zeros((4, 6, 3), np.uint8)
    img[..., 0] = 10  # B
    img[..., 1] = 20  # G
    img[..., 2] = 30  # R
    return img


# --- Mapping -----------------------------------------------------------------


def test_detect_maps_the_mesh_to_box_nose_eyes_and_landmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch, [_NativeFace(mesh(), np.eye(4))])
    (face,) = MediaPipeFaceDetector().detect(image(), 0.0)

    assert face.box == pytest.approx((0.3, 0.2, 0.6, 0.7))
    assert face.nose == pytest.approx((0.45, 0.5))
    assert face.eyes is not None
    right, left = face.eyes
    assert right == pytest.approx((0.38, 0.35))  # the subject's right eye first
    assert left == pytest.approx((0.52, 0.36))
    assert face.score is None
    assert face.orientation == pytest.approx((0.0, 0.0, 0.0))
    assert face.landmarks is not None
    assert face.landmarks.shape == (478, 3) and face.landmarks.dtype == np.float32


def test_box_is_clamped_to_the_image(monkeypatch: pytest.MonkeyPatch) -> None:
    pts = mesh()
    pts[30] = (-0.2, -0.1, 0.0)  # face partly off the top-left
    pts[31] = (1.3, 1.05, 0.0)  # and off the bottom-right
    install_fake(monkeypatch, [_NativeFace(pts, None)])
    (face,) = MediaPipeFaceDetector().detect(image(), 0.0)
    assert face.box == (0.0, 0.0, 1.0, 1.0)
    assert face.orientation is None  # no matrix, no fitted head


def test_every_face_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    second = mesh()
    second[:, 0] += 0.3
    install_fake(monkeypatch, [_NativeFace(mesh(), None), _NativeFace(second, None)])
    faces = MediaPipeFaceDetector(max_faces=2).detect(image(), 0.0)
    assert [f.box[0] for f in faces] == pytest.approx([0.3, 0.6])


def test_no_face_gives_an_empty_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, [])
    assert MediaPipeFaceDetector().detect(image(), 0.0) == ()


def test_short_mesh_has_no_eyes(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, [_NativeFace(mesh(468), None)])  # no iris points
    (face,) = MediaPipeFaceDetector().detect(image(), 0.0)
    assert face.eyes is None
    assert face.nose == pytest.approx((0.45, 0.5))


# --- Orientation -------------------------------------------------------------


def test_identity_matrix_is_a_frontal_upright_face() -> None:
    assert orientation_from_matrix(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize("deg", [15.0, -15.0])
def test_pure_roll(deg: float) -> None:
    t = math.radians(deg)
    # Clockwise in the image is clockwise seen from the camera: -z in its frame.
    roll, pitch, yaw = orientation_from_matrix(pose(rz(-t)))
    assert (roll, pitch, yaw) == pytest.approx((t, 0.0, 0.0), abs=1e-6)


@pytest.mark.parametrize("deg", [15.0, -15.0])
def test_pure_pitch(deg: float) -> None:
    t = math.radians(deg)
    roll, pitch, yaw = orientation_from_matrix(pose(rx(t)))
    assert (roll, pitch, yaw) == pytest.approx((0.0, t, 0.0), abs=1e-6)


@pytest.mark.parametrize("deg", [15.0, -15.0])
def test_pure_yaw(deg: float) -> None:
    t = math.radians(deg)
    roll, pitch, yaw = orientation_from_matrix(pose(ry(t)))
    assert (roll, pitch, yaw) == pytest.approx((0.0, 0.0, t), abs=1e-6)


def test_scale_does_not_change_the_angles() -> None:
    r = rz(0.2) @ ry(-0.3) @ rx(0.1)
    assert orientation_from_matrix(pose(r, scale=3.5)) == pytest.approx(
        orientation_from_matrix(pose(r)), abs=1e-9
    )


def test_roll_agrees_in_sign_with_the_eye_line(monkeypatch: pytest.MonkeyPatch) -> None:
    # Right eye (image left) higher than the left eye: the eye line turned clockwise.
    pts = mesh()
    pts[RIGHT_IRIS] = (0.38, 0.30, 0.0)
    pts[LEFT_IRIS] = (0.52, 0.40, 0.0)
    t = math.atan2(0.40 - 0.30, 0.52 - 0.38)
    install_fake(monkeypatch, [_NativeFace(pts, pose(rz(-t)))])
    (face,) = MediaPipeFaceDetector().detect(image(), 0.0)
    assert face.eyes is not None and face.orientation is not None
    (rx_, ry_), (lx, ly) = face.eyes
    assert face.orientation[0] == pytest.approx(
        math.atan2(ly - ry_, lx - rx_), abs=1e-6
    )


# --- Input handling ----------------------------------------------------------


def test_image_is_handed_over_as_contiguous_rgb_and_caller_array_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    img = image()
    before = img.copy()
    MediaPipeFaceDetector().detect(img, 0.0)

    rgb, _ = fake.calls[0]
    assert rgb.flags.c_contiguous
    assert tuple(rgb[0, 0]) == (30, 20, 10)
    assert not np.shares_memory(rgb, img)
    assert np.array_equal(img, before)


def test_video_timestamps_are_strictly_increasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    det = MediaPipeFaceDetector()
    for ts in (1.0, 1.0, 0.5, 1.2345, 2.0):
        det.detect(image(), ts)
    assert [ts_ms for _, ts_ms in fake.calls] == [1000, 1001, 1002, 1234, 2000]


# --- Construction and lifecycle ----------------------------------------------


def test_max_faces_below_one_is_rejected_before_opening_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_open(*args: object) -> None:
        raise AssertionError("validation must happen before the landmarker is opened")

    monkeypatch.setattr(face_module, "_open_landmarker", must_not_open)
    with pytest.raises(ValueError, match="max_faces"):
        MediaPipeFaceDetector(max_faces=0)


def test_max_faces_is_passed_to_the_landmarker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def open_landmarker(model_path, max_faces, *confidences) -> FakeLandmarker:
        seen.append(max_faces)
        return FakeLandmarker([])

    monkeypatch.setattr(face_module, "_open_landmarker", open_landmarker)
    MediaPipeFaceDetector(max_faces=4)
    assert seen == [4]


def test_close_is_idempotent_and_detect_after_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    det = MediaPipeFaceDetector()
    det.close()
    det.close()
    assert fake.close_count == 1
    with pytest.raises(RuntimeError):
        det.detect(image(), 0.0)
