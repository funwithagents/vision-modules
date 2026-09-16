import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_modules.stream import OpenCVSource, StreamProvider

# --- Test helpers ------------------------------------------------------------


class SteppedSource:
    """Yields one image per allow(); read() blocks until allowed; close() unblocks it."""

    def __init__(self, images: list[np.ndarray]):
        self._images = list(images)
        self._gate = threading.Semaphore(0)
        self.closed = True  # until open()
        self.opens = 0
        self.reads = 0

    def allow(self, n: int = 1):
        [self._gate.release() for _ in range(n)]

    def open(self):
        self.opens += 1
        self.closed = False
        self._gate = threading.Semaphore(0)  # drop the release() close() added

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
    assert src.opens == 0
    assert src.reads == 0
    assert sp.latest() is None
    assert sp.ended is False


def test_start_opens_the_source() -> None:
    src = SteppedSource([make_image()])
    sp = StreamProvider(src)
    sp.start()
    try:
        assert src.opens == 1
        assert src.closed is False
    finally:
        sp.stop()


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


def test_stop_does_not_mark_the_stream_as_ended() -> None:
    src = SteppedSource([make_image()])
    sp = StreamProvider(src)
    sp.start()
    sp.stop()  # interrupts a blocked read(): that is a stop, not an end of source
    assert sp.ended is False
    assert src.reads == 1  # the interrupted read did return (None)


def test_restart_reopens_the_source_and_keeps_frame_ids_monotonic() -> None:
    src = SteppedSource([make_image(), make_image()])
    sp = StreamProvider(src)

    def frame_id() -> int | None:
        f = sp.latest()
        return f.frame_id if f is not None else None

    sp.start()
    src.allow()
    wait_until(lambda: frame_id() == 1)
    sp.stop()
    assert src.closed is True and src.opens == 1
    assert frame_id() == 1  # last frame survives the stop

    sp.start()
    try:
        assert src.opens == 2 and src.closed is False
        assert sp.ended is False
        src.allow()
        wait_until(lambda: frame_id() == 2)  # continues counting, no reset to 1
    finally:
        sp.stop()
    assert not any(t.name == "StreamProvider" for t in threading.enumerate())


def test_restart_after_the_source_ended_clears_ended() -> None:
    src = SteppedSource([make_image()])
    sp = StreamProvider(src)
    sp.start()
    src.allow(2)
    wait_until(lambda: sp.ended)
    sp.stop()
    sp.start()
    try:
        assert sp.ended is False
    finally:
        sp.stop()


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
    assert not any(t.name == "StreamProvider" for t in threading.enumerate())


# --- OpenCVSource on a generated video file (no camera, no network) -----------


def write_clip(path: Path, n_frames: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened(), "cv2 cannot write MJPG here"
    for i in range(n_frames):
        writer.write(np.full((24, 32, 3), i * 40, np.uint8))
    writer.release()


def test_opencv_source_open_read_close_reopen(tmp_path: Path) -> None:
    write_clip(tmp_path / "clip.avi", 3)
    src = OpenCVSource(str(tmp_path / "clip.avi"))
    assert src.read() is None  # constructor does not open
    src.open()
    first = src.read()
    assert first is not None and first.shape == (24, 32, 3)
    src.close()
    assert src.read() is None  # closed: no crash, no frame
    src.close()  # idempotent
    src.open()  # re-open replays from the start
    again = src.read()
    assert again is not None and np.array_equal(again, first)
    src.close()


def test_provider_reads_a_file_to_the_end_and_replays_it_on_restart(
    tmp_path: Path,
) -> None:
    write_clip(tmp_path / "clip.avi", 3)
    sp = StreamProvider(str(tmp_path / "clip.avi"))
    sp.start()
    try:
        wait_until(lambda: sp.ended)
        f = sp.latest()
        assert f is not None and f.frame_id == 3
        assert f.image.shape == (24, 32, 3) and f.image.dtype == np.uint8
    finally:
        sp.stop()
    sp.start()
    try:
        wait_until(lambda: sp.ended)
        f = sp.latest()
        assert f is not None and f.frame_id == 6  # 3 more frames, ids continue
    finally:
        sp.stop()
