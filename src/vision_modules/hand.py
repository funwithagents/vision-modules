"""Hand — shared hand stage: detect and crop hands once per frame.

Governed by specs/hand.md.
"""

import logging
import os
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

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


MATCH_IOU = 0.3  # a candidate must overlap a previously published box this much to be its incumbent
Rank = Literal["area", "score"]


def box_area(box: tuple[float, float, float, float]) -> float:
    """Area of a normalized (x0, y0, x1, y1) box; 0.0 for an inverted box."""
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection over union of two normalized boxes; 0.0 when they don't overlap."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def select_hands(
    candidates: Sequence[DetectedHand],
    previous: Sequence[DetectedHand],
    max_hands: int,
    rank: Rank,
    hysteresis: float,
) -> tuple[DetectedHand, ...]:
    """Pick at most `max_hands` of `candidates`, slot-stable against `previous`.

    See specs/hand.md "Selection: rank, hysteresis, slot stability".
    """
    # 1. rank key per candidate
    keys = [box_area(d.box) if rank == "area" else d.score for d in candidates]

    # 2. match candidates to previously published slots by IoU, best pairs first
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(previous):
        for ci, c in enumerate(candidates):
            iou = box_iou(p.box, c.box)
            if iou >= MATCH_IOU:
                pairs.append((iou, pi, ci))
    pairs.sort(key=lambda t: t[0], reverse=True)
    slot_of: dict[int, int] = {}  # candidate index -> previous slot index
    used_slots: set[int] = set()
    for _, pi, ci in pairs:
        if pi in used_slots or ci in slot_of:
            continue
        slot_of[ci] = pi
        used_slots.add(pi)

    # 3. hysteresis: incumbents get a boost
    boosted = [
        k * (1.0 + hysteresis) if ci in slot_of else k for ci, k in enumerate(keys)
    ]

    # 4. choose: highest boosted key first; sorted() is stable, so ties keep detector order
    order = sorted(range(len(candidates)), key=lambda ci: boosted[ci], reverse=True)
    kept = order[:max_hands]

    # 5. order: incumbents in their previous slot order, then newcomers in rank order
    incumbents = sorted(
        (ci for ci in kept if ci in slot_of), key=lambda ci: slot_of[ci]
    )
    newcomers = [ci for ci in kept if ci not in slot_of]
    return tuple(candidates[ci] for ci in incumbents + newcomers)


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
        max_hands: int = 2,
        rank: Rank = "area",
        hysteresis: float = 0.2,
        detector: HandDetector | None = None,
    ) -> None:
        if max_hands < 1:
            raise ValueError(f"max_hands must be >= 1, got {max_hands}")
        if hysteresis < 0:
            raise ValueError(f"hysteresis must be >= 0, got {hysteresis}")
        if rank not in ("area", "score"):
            raise ValueError(f"rank must be 'area' or 'score', got {rank!r}")
        super().__init__(provider, target_fps)
        self.pad = pad
        self.max_hands = max_hands
        self.rank: Rank = rank
        self.hysteresis = hysteresis
        self._borrowed = detector  # caller's: used, never closed here
        self._owned: HandDetector | None = (
            None  # ours: created per run, closed in close()
        )
        self._previous: tuple[
            DetectedHand, ...
        ] = ()  # last published, worker thread only

    def _detector(self) -> HandDetector:
        if self._borrowed is not None:
            return self._borrowed
        if self._owned is None:
            self._owned = MediaPipeHandDetector(num_hands=self.max_hands)
        return self._owned

    def process(self, item: Frame) -> HandResult:
        frame = item
        found = self._detector().detect(frame.image, frame.ts)
        selected = select_hands(
            found, self._previous, self.max_hands, self.rank, self.hysteresis
        )
        self._previous = selected
        height, width = frame.image.shape[:2]
        hands: list[Hand] = []
        for d in selected:
            x0, y0, x1, y1 = padded_box(d.box, width, height, self.pad)
            crop = frame.image[y0:y1, x0:x1].copy() if x1 > x0 and y1 > y0 else None
            hands.append(Hand((x0, y0, x1, y1), crop, d.score))
        return HandResult(
            frame.frame_id, frame.ts, present=bool(hands), hands=tuple(hands)
        )

    def close(self) -> None:
        self._previous = ()  # a restarted run starts with no incumbents
        if self._owned is not None:
            self._owned.close()
            self._owned = None  # a restarted run creates a fresh one


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
