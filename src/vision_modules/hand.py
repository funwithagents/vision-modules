"""Hand — shared hand stage: detect and crop hands once per frame.

Governed by specs/hand.md.
"""

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from vision_modules.pipeline import Result, Stage, Upstream
from vision_modules.stream import Frame

log = logging.getLogger(__name__)

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class DetectedHand:
    box: tuple[
        float, float, float, float
    ]  # (x0, y0, x1, y1) normalized to [0, 1], UNPADDED
    score: float


class HandDetector(Protocol):
    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, eq=False)
class Hand:
    bbox: tuple[int, int, int, int]  # pixels, padded + clamped
    crop: np.ndarray | None  # independent copy; None only for a zero-area box
    score: float


@dataclass(frozen=True, eq=False)
class HandResult(Result):
    hands: tuple[Hand, ...]

    @property
    def first(self) -> Hand | None:
        return self.hands[0] if self.hands else None


def padded_box(
    box: tuple[float, float, float, float], width: int, height: int, pad: float
) -> tuple[int, int, int, int]:
    """Normalized unpadded box -> padded, clamped pixel box. Pure function."""
    x0, y0, x1, y1 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    bw, bh = x1 - x0, y1 - y0
    x0 -= pad * bw
    x1 += pad * bw
    y0 -= pad * bh
    y1 += pad * bh
    return (int(max(0, x0)), int(max(0, y0)), int(min(width, x1)), int(min(height, y1)))


class HandStage(Stage[Frame, HandResult]):
    def __init__(
        self,
        provider: Upstream[Frame],
        target_fps: float | None = 30,
        pad: float = 0.35,
        max_hands: int = 1,
        detector: HandDetector | None = None,
    ) -> None:
        super().__init__(provider, target_fps)
        self.pad = pad
        self.max_hands = max_hands
        self._given = detector
        self._detector: HandDetector | None = None

    def process(self, item: Frame) -> HandResult:
        frame = item
        if self._detector is None:
            self._detector = self._given or MediaPipeHandDetector(
                num_hands=self.max_hands
            )
        found = self._detector.detect(frame.image, frame.ts)
        found = tuple(sorted(found, key=lambda d: d.score, reverse=True))[
            : self.max_hands
        ]
        height, width = frame.image.shape[:2]
        hands: list[Hand] = []
        for d in found:
            x0, y0, x1, y1 = padded_box(d.box, width, height, self.pad)
            crop = frame.image[y0:y1, x0:x1].copy() if x1 > x0 and y1 > y0 else None
            hands.append(Hand((x0, y0, x1, y1), crop, d.score))
        return HandResult(
            frame.frame_id, frame.ts, present=bool(hands), hands=tuple(hands)
        )

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()


def default_model_path() -> Path:
    cache_dir = os.environ.get("VISION_MODULES_CACHE")
    base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "vision-modules"
    return base / "hand_landmarker.task"


def ensure_model(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    log.info("downloading hand landmarker model to %s", path)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


class MediaPipeHandDetector:
    DEFAULT_MODEL_URL = DEFAULT_MODEL_URL

    def __init__(
        self,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        path = Path(model_path) if model_path is not None else default_model_path()
        ensure_model(path, self.DEFAULT_MODEL_URL)

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp = mp
        self._lm = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedHand, ...]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(ts * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        res = self._lm.detect_for_video(mp_img, ts_ms)

        hands: list[DetectedHand] = []
        for i, landmarks in enumerate(res.hand_landmarks):
            xs = [p.x for p in landmarks]
            ys = [p.y for p in landmarks]
            box = (
                min(max(min(xs), 0.0), 1.0),
                min(max(min(ys), 0.0), 1.0),
                min(max(max(xs), 0.0), 1.0),
                min(max(max(ys), 0.0), 1.0),
            )
            score = res.handedness[i][0].score
            hands.append(DetectedHand(box, score))
        return tuple(hands)

    def close(self) -> None:
        self._lm.close()
