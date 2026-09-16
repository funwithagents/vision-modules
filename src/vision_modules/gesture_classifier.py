"""Gesture classifier — named gestures from the hand crop with an image model.

Governed by specs/gesture_classifier.md.
"""

from dataclasses import dataclass
from typing import Protocol, cast

import cv2
import numpy as np

from vision_modules.hand import HandResult
from vision_modules.pipeline import Module, Result, Upstream


@dataclass(frozen=True)
class HandGesture:
    label: (
        str | None
    )  # top class if its score >= threshold, else None; None when no crop
    confidence: float  # top score, 0.0 when no crop
    scores: dict[str, float]  # every class -> score; {} when no crop


@dataclass(frozen=True, eq=False)
class Gesture(Result):
    hands: tuple[HandGesture, ...]  # index-aligned with HandResult.hands

    @property
    def first(self) -> HandGesture | None:
        return self.hands[0] if self.hands else None


class ImageClassifier(Protocol):
    @property
    def labels(self) -> tuple[str, ...]: ...
    def classify(self, image_bgr: np.ndarray) -> dict[str, float]: ...
    def close(self) -> None: ...


def select_device(preferred: str | None = None) -> str:
    """preferred if given; else 'mps' if torch.backends.mps.is_available(), else 'cuda' if torch.cuda.is_available(), else 'cpu'."""
    if preferred is not None:
        return preferred
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class GestureClassifier(Module[HandResult, Gesture]):
    def __init__(
        self,
        hand_stage: Upstream[HandResult],
        target_fps: float | None = 5,
        threshold: float = 0.55,
        classifier: ImageClassifier | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(hand_stage, target_fps)
        self.threshold = threshold
        self._borrowed = classifier  # caller's: used, never closed here
        self._device = device
        self._owned: ImageClassifier | None = (
            None  # ours: loaded per run, closed in close()
        )

    def _classifier(self) -> ImageClassifier:
        if self._borrowed is not None:
            return self._borrowed
        if self._owned is None:
            self._owned = HaGRIDViTClassifier(device=select_device(self._device))
        return self._owned

    def process(self, item: HandResult) -> Gesture:
        hands = item
        clf = self._classifier()
        out: list[HandGesture] = []
        for hand in hands.hands:
            if hand.crop is None:
                out.append(HandGesture(None, 0.0, {}))
                continue
            scores = clf.classify(hand.crop)
            label, conf = max(scores.items(), key=lambda kv: kv[1])
            out.append(
                HandGesture(label if conf >= self.threshold else None, conf, scores)
            )
        return Gesture(hands.frame_id, hands.ts, present=bool(out), hands=tuple(out))

    def close(self) -> None:
        if self._owned is not None:
            self._owned.close()
            self._owned = None  # a restarted run loads a fresh one


class HaGRIDViTClassifier:
    MODEL_ID = "dima806/hand_gestures_image_detection"

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None:
        from transformers import pipeline as hf_pipeline

        self._pipe = hf_pipeline("image-classification", model=model_id, device=device)
        id2label = cast(dict[int, str], self._pipe.model.config.id2label)
        self._labels = tuple(id2label[i] for i in sorted(id2label))

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def classify(self, image_bgr: np.ndarray) -> dict[str, float]:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        preds = self._pipe(pil, top_k=len(self._labels))
        return {p["label"]: float(p["score"]) for p in preds}

    def close(self) -> None:
        del self._pipe
