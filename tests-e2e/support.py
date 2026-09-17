"""Helpers for the opt-in live tier.

A live test needs real credentials. Rather than fail when they're absent, a test
calls `require_env(...)` up front so it *skips* cleanly — you only exercise the
services you hold keys for, and a contributor (or CI) with none is never broken.
"""

import os
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
NO_HAND_IMAGE = FIXTURES_DIR / "no_hand.jpg"
TWO_HANDS_IMAGE = FIXTURES_DIR / "two_hands.jpg"


def require_env(name: str) -> str:
    """Return env var `name`, or skip the calling test if it's unset/empty."""
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set; skipping live test")
    return value


def labeled_hand_fixtures() -> list[tuple[str, Path]]:
    """(label, path) for every tests-e2e/fixtures/hand_<label>.jpg.

    <label> is the HaGRID class the photo was taken to show — ground truth for
    tests that check the gesture classifier's actual prediction, not just that
    it produced *a* label.
    """
    return sorted(
        (path.stem.removeprefix("hand_"), path)
        for path in FIXTURES_DIR.glob("hand_*.jpg")
    )


def hand_image_path() -> Path:
    """Path to a real photo with one clearly visible hand (gesture doesn't matter).

    `VISION_MODULES_HAND_IMAGE` overrides; otherwise the first bundled fixture is
    used. Skips if neither is available, so a checkout without fixtures still
    skips cleanly instead of failing.
    """
    override = os.environ.get("VISION_MODULES_HAND_IMAGE")
    if override:
        return Path(override)
    fixtures = labeled_hand_fixtures()
    if fixtures:
        return fixtures[0][1]
    pytest.skip(
        "no hand image available: add tests-e2e/fixtures/hand_<gesture>.jpg or "
        "set VISION_MODULES_HAND_IMAGE to a photo with one clearly visible hand"
    )


def no_hand_image_path() -> Path:
    """Path to a real photo with no hand in frame, for the negative path.

    Skips if the bundled fixture isn't present.
    """
    if NO_HAND_IMAGE.exists():
        return NO_HAND_IMAGE
    pytest.skip("no negative fixture available: add tests-e2e/fixtures/no_hand.jpg")


def two_hands_image_path() -> Path:
    """Path to a real photo with two clearly visible hands. Skips if absent."""
    if TWO_HANDS_IMAGE.exists():
        return TWO_HANDS_IMAGE
    pytest.skip("no two-hand fixture available: add tests-e2e/fixtures/two_hands.jpg")
