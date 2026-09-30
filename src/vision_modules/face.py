"""Face — the `FaceDetector` seam and its MediaPipe backend.

Governed by specs/face.md.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vision_modules.hand import default_model_path, ensure_model, video_ts_ms

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"
EXTRA_HINT = 'pip install "vision-modules[face-mediapipe]"'

# MediaPipe face mesh indices (478 points, irises included).
NOSE_TIP = 1
RIGHT_IRIS = 468  # the subject's right eye: the image's left in an unmirrored image
LEFT_IRIS = 473


@dataclass(frozen=True, eq=False)
class DetectedFace:
    """One detected face; every coordinate normalised to the image given."""

    box: tuple[float, float, float, float]  # (x0, y0, x1, y1), unpadded, clamped
    score: float | None
    nose: tuple[float, float] | None
    eyes: tuple[tuple[float, float], tuple[float, float]] | None  # (right, left)
    orientation: tuple[float, float, float] | None  # (roll, pitch, yaw), radians
    landmarks: np.ndarray | None  # (N, 3) float32


class FaceDetector(Protocol):
    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedFace, ...]: ...
    def close(self) -> None: ...


def orientation_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """`(roll, pitch, yaw)` in the spec's image convention from MediaPipe's
    facial transformation matrix.

    MediaPipe's camera frame is OpenGL-like (x right, y up, z toward the viewer).
    The rotation is decomposed as `Rz(a) · Ry(b) · Rx(c)`: a counter-clockwise
    turn about +z is counter-clockwise in the image (roll = -a); +x brings the
    top of the head toward the camera, chin down (pitch = c); +y turns the face
    toward +x, the image's right (yaw = b).
    """
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    r = r / np.linalg.norm(r, axis=0)  # drop any scale
    b = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    c = math.atan2(r[2, 1], r[2, 2])
    a = math.atan2(r[1, 0], r[0, 0])
    return (-a, c, b)


@dataclass(frozen=True, eq=False)
class _NativeFace:
    landmarks: np.ndarray  # (N, 3) float32
    matrix: np.ndarray | None  # (4, 4)


class _Landmarker(Protocol):
    def detect(self, rgb: np.ndarray, ts_ms: int) -> list[_NativeFace]: ...
    def close(self) -> None: ...


class _MediaPipeLandmarker:
    def __init__(self, mp: Any, landmarker: Any) -> None:
        self._mp = mp
        self._lm = landmarker

    def detect(self, rgb: np.ndarray, ts_ms: int) -> list[_NativeFace]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(image, ts_ms)
        matrices = res.facial_transformation_matrixes or []
        faces: list[_NativeFace] = []
        for i, mesh in enumerate(res.face_landmarks):
            pts = np.array([(p.x, p.y, p.z) for p in mesh], dtype=np.float32)
            matrix = (
                np.asarray(matrices[i], dtype=np.float64) if i < len(matrices) else None
            )
            faces.append(_NativeFace(pts.reshape(-1, 3), matrix))
        return faces

    def close(self) -> None:
        self._lm.close()


def _open_landmarker(
    model_path: str | Path | None,
    max_faces: int,
    min_face_detection_confidence: float,
    min_face_presence_confidence: float,
    min_tracking_confidence: float,
) -> _Landmarker:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as e:
        raise ImportError(f"MediaPipeFaceDetector needs mediapipe: {EXTRA_HINT}") from e

    path = (
        Path(model_path)
        if model_path is not None
        else default_model_path(MODEL_FILENAME)
    )
    ensure_model(path, DEFAULT_MODEL_URL)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=max_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return _MediaPipeLandmarker(mp, vision.FaceLandmarker.create_from_options(options))


def _point(landmarks: np.ndarray, i: int) -> tuple[float, float]:
    return (float(landmarks[i, 0]), float(landmarks[i, 1]))


def _to_detected(native: _NativeFace) -> DetectedFace:
    pts = native.landmarks.astype(np.float32, copy=False)
    xs = np.clip(pts[:, 0], 0.0, 1.0)
    ys = np.clip(pts[:, 1], 0.0, 1.0)
    box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    n = len(pts)
    nose = _point(pts, NOSE_TIP) if n > NOSE_TIP else None
    eyes = (
        (_point(pts, RIGHT_IRIS), _point(pts, LEFT_IRIS))
        if n > max(RIGHT_IRIS, LEFT_IRIS)
        else None
    )
    orientation = (
        orientation_from_matrix(native.matrix) if native.matrix is not None else None
    )
    return DetectedFace(box, None, nose, eyes, orientation, pts)


class MediaPipeFaceDetector:
    """`FaceDetector` over MediaPipe's Tasks `FaceLandmarker` in VIDEO mode.

    Not thread-safe: one caller at a time; successive calls may come from
    different threads. One instance per video (it tracks between calls).
    """

    DEFAULT_MODEL_URL = DEFAULT_MODEL_URL

    def __init__(
        self,
        max_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ) -> None:
        if max_faces < 1:
            raise ValueError(f"max_faces must be >= 1, got {max_faces}")
        self._lm = _open_landmarker(
            model_path,
            max_faces,
            min_face_detection_confidence,
            min_face_presence_confidence,
            min_tracking_confidence,
        )
        self._last_ts_ms = -1
        self._closed = False

    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedFace, ...]:
        if self._closed:
            raise RuntimeError("MediaPipeFaceDetector.detect() after close()")
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])  # BGR -> RGB, own buffer
        ts_ms = video_ts_ms(ts, self._last_ts_ms)
        self._last_ts_ms = ts_ms
        return tuple(_to_detected(f) for f in self._lm.detect(rgb, ts_ms))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lm.close()
