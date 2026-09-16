import threading
import time

import numpy as np
import pytest

from vision_modules.stream import StreamProvider

# --- Test helpers ------------------------------------------------------------


class SteppedSource:
    """Yields one image per allow(); read() blocks until allowed; close() unblocks it."""

    def __init__(self, images: list[np.ndarray]):
        self._images = list(images)
        self._gate = threading.Semaphore(0)
        self.closed = False
        self.reads = 0

    def allow(self, n: int = 1):
        [self._gate.release() for _ in range(n)]

    def read(self):
        self._gate.acquire()
        self.reads += 1
        if self.closed or not self._images:
            return None
        return self._images.pop(0)

    def close(self):
        self.closed = True
        self._gate.release()  # release a read blocked in acquire()


def wait_until(pred: object, timeout: float = 2.0) -> None:
    assert callable(pred)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    assert pred(), "condition not met within timeout"


def make_image() -> np.ndarray:
    img = np.zeros((4, 6, 3), np.uint8)
    img[:, 0] = 255  # white left column
    return img


# --- Tests -------------------------------------------------------------------


def test_constructor_does_not_touch_the_source() -> None:
    src = SteppedSource([make_image()])
    sp = StreamProvider(src)
    assert src.reads == 0
    assert sp.latest() is None
    assert sp.ended is False


def test_frames_get_increasing_ids_and_monotonic_timestamps() -> None:
    src = SteppedSource([make_image(), make_image()])
    sp = StreamProvider(src)

    def frame_id() -> int | None:
        f = sp.latest()
        return f.frame_id if f is not None else None

    sp.start()
    try:
        src.allow()
        wait_until(lambda: frame_id() == 1)
        f1 = sp.latest()
        assert f1 is not None

        src.allow()
        wait_until(lambda: frame_id() == 2)
        f2 = sp.latest()
        assert f2 is not None
        assert f2.ts > f1.ts
    finally:
        sp.stop()


def test_frame_image_is_the_source_array_not_a_copy() -> None:
    images = [make_image()]
    src = SteppedSource(images)
    sp = StreamProvider(src)
    sp.start()
    try:
        src.allow()
        wait_until(lambda: sp.latest() is not None)
        f = sp.latest()
        assert f is not None
        assert f.image is images[0]
    finally:
        sp.stop()


def test_earlier_frame_stays_valid_after_provider_advances() -> None:
    images = [make_image(), make_image()]
    original_copy_of_images_0 = images[0].copy()
    src = SteppedSource(images)
    sp = StreamProvider(src)
    sp.start()
    try:
        src.allow()
        wait_until(lambda: sp.latest() is not None)
        f1 = sp.latest()
        assert f1 is not None
        assert f1.frame_id == 1

        src.allow()
        wait_until(lambda: (sp.latest() or f1).frame_id == 2)

        assert f1.frame_id == 1
        assert f1.image is images[0]
        assert np.array_equal(f1.image, original_copy_of_images_0)
    finally:
        sp.stop()


def test_mirror_flips_horizontally_into_a_new_array() -> None:
    images = [make_image()]
    src = SteppedSource(images)
    sp = StreamProvider(src, mirror=True)
    sp.start()
    try:
        src.allow()
        wait_until(lambda: sp.latest() is not None)
        f = sp.latest()
        assert f is not None
        assert (f.image[:, -1] == 255).all()
        assert (f.image[:, 0] == 0).all()
        assert f.image is not images[0]
        assert (images[0][:, 0] == 255).all()
    finally:
        sp.stop()


def test_ended_becomes_true_when_source_is_exhausted() -> None:
    src = SteppedSource([make_image()])
    sp = StreamProvider(src)
    sp.start()
    try:
        src.allow(2)  # second read returns None
        wait_until(lambda: sp.ended)
        f = sp.latest()
        assert f is not None
        assert f.frame_id == 1
    finally:
        sp.stop()


def test_stop_joins_and_closes_the_source() -> None:
    src = SteppedSource([make_image()])
    sp = StreamProvider(src)
    sp.start()
    sp.stop()  # worker is blocked in read(); stop() must still return
    assert src.closed is True
    assert not any(t.name == "StreamProvider" for t in threading.enumerate())
    sp.stop()  # idempotent, no raise


def test_context_manager_starts_and_stops() -> None:
    src = SteppedSource([make_image()])
    with StreamProvider(src) as sp:
        src.allow()
        wait_until(lambda: sp.latest() is not None)
    assert src.closed


def test_int_source_that_cannot_open_raises_from_start() -> None:
    sp = StreamProvider(source=999999)
    with pytest.raises(RuntimeError):
        sp.start()
