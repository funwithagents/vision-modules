import time

import numpy as np
from support import require_env

from vision_modules.stream import StreamProvider


def test_webcam_publishes_bgr_frames() -> None:
    index = int(require_env("VISION_MODULES_CAMERA"))
    with StreamProvider(index) as sp:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and sp.latest() is None:
            time.sleep(0.05)
        f = sp.latest()
        assert f is not None, "no frame published within 5s"
        assert f.image.ndim == 3
        assert f.image.shape[2] == 3
        assert f.image.dtype == np.uint8
        # Nominal rate: webcams may report 0 (mapped to None), never negative.
        assert sp.source_fps is None or sp.source_fps > 0
        # Measured rate: the counter advances while frames keep arriving.
        before = sp.published_count
        assert before >= 1
        time.sleep(0.5)
        assert sp.published_count > before
