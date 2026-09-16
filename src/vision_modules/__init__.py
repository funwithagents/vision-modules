"""Vision Modules — modular perception over a video stream.

Public re-exports land here as modules are built.
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
    Hand,
    HandDetector,
    HandResult,
    HandStage,
    MediaPipeHandDetector,
)
from vision_modules.pipeline import LatestValue, Module, Pipeline, Result, Stage
from vision_modules.stream import Frame, FrameSource, OpenCVSource, StreamProvider

__all__ = [
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
    "select_device",
]
