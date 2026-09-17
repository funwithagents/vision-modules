"""Snapshot — save what a node last published or last consumed to a local file.

Pure functions over already-published values: call them from wherever you
read `latest()` / `last_input`, never from a worker. Governed by specs/snapshot.md.
"""

from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_modules.hand import HandResult
from vision_modules.pipeline import Stage
from vision_modules.stream import Frame


def _write(image: np.ndarray, path: str | PathLike[str]) -> Path:
    """Encode a BGR array to `path` by suffix; create parents; never swallow failure."""
    path = Path(path)
    if not cv2.haveImageWriter(str(path)):
        raise ValueError(f"unsupported image format {path.suffix!r} for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write image to {path}")
    return path


def save_frame(frame: Frame | None, path: str | PathLike[str]) -> Path:
    """Write `frame.image` (BGR, unconverted) to `path` and return it.

    Accepts `provider.latest()` directly: `None` (nothing published yet) raises
    `ValueError` rather than silently writing nothing.
    """
    if frame is None:
        raise ValueError("no frame published yet: nothing to save")
    return _write(frame.image, path)


def save_crops(
    result: HandResult | None, path: str | PathLike[str]
) -> tuple[Path | None, ...]:
    """Write one file per hand crop; return paths index-aligned with `result.hands`.

    Hand `i` is written to `path` with `_i` inserted before the suffix
    (`crop.png` -> `crop_0.png`, `crop_1.png`, ...). A hand without a crop
    (zero-area box) yields `None`; an empty result writes nothing and returns `()`.
    """
    if result is None:
        raise ValueError("no hand result published yet: nothing to save")
    path = Path(path)
    written: list[Path | None] = []
    for i, hand in enumerate(result.hands):
        if hand.crop is None:
            written.append(None)
            continue
        written.append(
            _write(hand.crop, path.with_name(f"{path.stem}_{i}{path.suffix}"))
        )
    return tuple(written)


def save_input(
    stage: Stage[Any, Any], path: str | PathLike[str]
) -> Path | tuple[Path | None, ...]:
    """Write the image(s) `stage` last handed to `process()` — its `last_input`.

    Dispatches on the input's type: a `Frame` (e.g. `HandStage`) is written like
    `save_frame`, a `HandResult` (e.g. `GestureClassifier`) like `save_crops`, and
    the matching return shape comes back. Nothing processed yet raises
    `ValueError`; an input type with no pixels this module knows raises `TypeError`.
    """
    item = stage.last_input
    if item is None:
        raise ValueError(
            f"{stage.name} has not processed anything yet: nothing to save"
        )
    if isinstance(item, Frame):
        return save_frame(item, path)
    if isinstance(item, HandResult):
        return save_crops(item, path)
    raise TypeError(
        f"{stage.name}: don't know how to save an input of type {type(item).__name__}"
    )
