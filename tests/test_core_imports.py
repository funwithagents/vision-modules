"""The core runs on numpy alone (specs/project.md): with every optional package
blocked, the package imports, a stream feeds a custom stage, and the
OpenCV-backed paths fail with an error naming the `opencv` extra."""

import subprocess
import sys
import textwrap

CORE_ONLY_SCRIPT = textwrap.dedent(
    """
    import sys
    import tempfile
    import time
    from dataclasses import dataclass
    from pathlib import Path

    for name in ("cv2", "mediapipe", "torch", "transformers", "PIL"):
        sys.modules[name] = None  # makes `import name` raise ImportError

    import numpy as np

    import vision_modules
    from vision_modules import Frame, Pipeline, Result, Stage, StreamProvider, save_frame

    for name in vision_modules.__all__:  # every root export resolves
        getattr(vision_modules, name)


    class ListSource:
        def __init__(self, images):
            self._images = list(images)

        def open(self):
            pass

        def read(self):
            return self._images.pop(0) if self._images else None

        def close(self):
            pass

        def fps(self):
            return None


    @dataclass(frozen=True)
    class Total(Result):
        total: int


    class Summer(Stage[Frame, Total]):
        def process(self, item):
            return Total(item.frame_id, item.ts, True, int(item.image.sum()))


    img = np.zeros((4, 6, 3), np.uint8)
    img[:, 0] = 7  # left column only, so the mirror is visible
    provider = StreamProvider(ListSource([img]), mirror=True)
    summer = Summer(provider, target_fps=None)
    with Pipeline([provider, summer]):
        deadline = time.monotonic() + 5
        while summer.latest() is None and time.monotonic() < deadline:
            time.sleep(0.005)
    frame = provider.latest()
    assert frame is not None and provider.ended
    assert np.array_equal(frame.image, img[:, ::-1])
    assert frame.image.flags.c_contiguous
    result = summer.latest()
    assert result is not None and result.total == int(img.sum())

    camera = StreamProvider(0)  # constructing needs no OpenCV
    try:
        camera.start()
    except ImportError as e:
        assert "vision-modules[opencv]" in str(e), e
    else:
        raise AssertionError("start() on a camera index should need the opencv extra")

    out = Path(tempfile.mkdtemp()) / "sub" / "x.png"
    try:
        save_frame(frame, out)
    except ImportError as e:
        assert "vision-modules[opencv]" in str(e), e
    else:
        raise AssertionError("save_frame should need the opencv extra")
    assert not out.parent.exists()
    print("ok")
    """
)


def test_core_runs_with_only_numpy_installed() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", CORE_ONLY_SCRIPT],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
