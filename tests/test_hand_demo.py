"""Unit tests for the pure helpers of examples/hand_demo.py (on pytest's pythonpath)."""

import threading
import time

import hand_demo
import numpy as np
import pytest
from hand_demo import FpsMeter, PushFrameSource, summarize

from vision_modules import Gesture, Hand, HandGesture, HandResult

# --- FpsMeter ------------------------------------------------------------------


def test_fps_meter_needs_two_samples_then_reports_the_exact_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(hand_demo.time, "monotonic", lambda: clock[0])
    meter = FpsMeter()
    assert meter.sample(0) is None  # nothing to compare against yet
    clock[0] += 2.0
    assert meter.sample(10) == 5.0  # 10 publishes over 2 s
    clock[0] += 0.5
    assert meter.sample(10) == 0.0  # stalled stage reads 0, not None


# --- summarize -------------------------------------------------------------------


def _hand_result(n_hands: int) -> HandResult:
    hands = tuple(
        Hand((0, 0, 8, 8), np.zeros((8, 8, 3), np.uint8), 0.9) for _ in range(n_hands)
    )
    return HandResult(1, 1.0, present=bool(hands), hands=hands)


def _gesture(label: str | None, scores: dict[str, float]) -> Gesture:
    return Gesture(
        1, 1.0, present=True, hands=(HandGesture(label, max(scores.values()), scores),)
    )


def test_summarize_without_a_hand() -> None:
    assert summarize(None, None) == {"no hand": 1.0}
    assert summarize(_hand_result(0), _gesture("palm", {"palm": 1.0})) == {
        "no hand": 1.0
    }


def test_summarize_hand_seen_but_classifier_not_caught_up() -> None:
    assert summarize(_hand_result(1), None) == {"...": 1.0}
    empty = Gesture(1, 1.0, present=False, hands=())
    assert summarize(_hand_result(1), empty) == {"...": 1.0}


def test_summarize_below_threshold_shows_plain_scores() -> None:
    scores = {"fist": 0.4, "palm": 0.6}
    assert summarize(_hand_result(1), _gesture(None, scores)) == scores


def test_summarize_marks_the_validated_label() -> None:
    out = summarize(_hand_result(1), _gesture("palm", {"fist": 0.2, "palm": 0.8}))
    assert out == {"fist": 0.2, "✓ palm": 0.8}


# --- PushFrameSource ---------------------------------------------------------------


def _read_in_thread(src: PushFrameSource) -> tuple[threading.Thread, list[object]]:
    out: list[object] = []
    t = threading.Thread(target=lambda: out.append(src.read()), daemon=True)
    t.start()
    return t, out


def test_push_source_read_blocks_until_a_frame_is_pushed() -> None:
    src = PushFrameSource()
    src.open()
    t, out = _read_in_thread(src)
    time.sleep(0.05)
    assert t.is_alive()  # nothing pushed: still blocked
    img = np.zeros((2, 2, 3), np.uint8)
    src.push(img)
    t.join(2.0)
    assert out == [img]
    assert out[0] is img  # handed over by reference, no copy


def test_push_source_close_unblocks_read_and_open_restarts_it() -> None:
    src = PushFrameSource()
    src.open()
    t, out = _read_in_thread(src)
    src.close()
    t.join(2.0)
    assert out == [None]  # closed: the blocked read returned None
    assert src.read() is None  # and stays closed
    src.open()
    img = np.ones((2, 2, 3), np.uint8)
    src.push(img)
    assert src.read() is img  # restartable
