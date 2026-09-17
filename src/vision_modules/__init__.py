"""Vision Modules — modular perception over a video stream.

Public API: everything a consumer needs to wire the shipped graph, write a
custom stage (`Stage`/`Module`/`Upstream`/`Result`) or plug a custom backend
(`FrameSource`, `HandDetector`/`DetectedHand`, `ImageClassifier`), plus
`save_frame`/`save_crops`/`save_input` to snapshot what a node last published or
last consumed to disk.
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
    select_hands,
)
from vision_modules.pipeline import (
    LatestValue,
    Module,
    Pipeline,
    Result,
    Stage,
    Upstream,
)
from vision_modules.snapshot import save_crops, save_frame, save_input
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
    "save_crops",
    "save_frame",
    "save_input",
    "select_device",
    "select_hands",
]
