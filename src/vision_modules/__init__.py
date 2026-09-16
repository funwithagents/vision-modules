"""Vision Modules — modular perception over a video stream.

Public API: everything a consumer needs to wire the shipped graph, write a
custom stage (`Stage`/`Module`/`Upstream`/`Result`) or plug a custom backend
(`FrameSource`, `HandDetector`/`DetectedHand`, `ImageClassifier`).
"""

from vision_modules.gesture_classifier import (
    Gesture,
    GestureClassifier,
    HaGRIDViTClassifier,
    HandGesture,
    ImageClassifier,
    select_device,
)
from vision_modules.hand import (
    DetectedHand,
    Hand,
    HandDetector,
    HandResult,
    HandStage,
    MediaPipeHandDetector,
)
from vision_modules.pipeline import (
    LatestValue,
    Module,
    Pipeline,
    Result,
    Stage,
    Upstream,
)
from vision_modules.stream import Frame, FrameSource, OpenCVSource, StreamProvider

__all__ = [
    "DetectedHand",
    "Frame",
    "FrameSource",
    "Gesture",
    "GestureClassifier",
    "HaGRIDViTClassifier",
    "Hand",
    "HandDetector",
    "HandGesture",
    "HandResult",
    "HandStage",
    "ImageClassifier",
    "LatestValue",
    "MediaPipeHandDetector",
    "Module",
    "OpenCVSource",
    "Pipeline",
    "Result",
    "Stage",
    "StreamProvider",
    "Upstream",
    "select_device",
]
