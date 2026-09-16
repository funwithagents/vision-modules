import numpy as np

from vision_modules.gesture_classifier import HaGRIDViTClassifier


def test_hagrid_model_loads_and_scores_every_class() -> None:
    clf = HaGRIDViTClassifier(
        device="cpu"
    )  # cpu: deterministic and available everywhere
    try:
        assert len(clf.labels) == 18 and {"palm", "stop", "fist"} <= set(clf.labels)
        scores = clf.classify(np.full((224, 224, 3), 127, np.uint8))
        assert set(scores) == set(clf.labels)
        assert abs(sum(scores.values()) - 1.0) < 1e-3
    finally:
        clf.close()
