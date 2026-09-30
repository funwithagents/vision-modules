---
code:
  - src/vision_modules/face.py
tests:
  - tests/test_face.py
  - tests-e2e/test_face_live.py
---

# Face

**Status:** Implemented

## Purpose

The face front-end: a **`FaceDetector` seam** that finds the faces in an image and says where each one is — a box, the nose and the eyes, a fitted head orientation, and the full landmarks when the backend has them — and its first backend, **`MediaPipeFaceDetector`**, over MediaPipe's `FaceLandmarker`. It is what every face-based consumer builds on: a future `FaceStage` in this library (open question 1), a recogniser (open question 2), and a host application outside the library that runs the detector in its own detection loop over its own camera. The seam is the contract such a consumer depends on; this spec fixes it.

It is a detector, not a stage: no thread, no `latest()`, no selection. Like [hand.md](hand.md)'s `HandDetector`, it is called with one image and a timestamp and returns what it found; the caller owns the loop. Installed through the **`face-mediapipe`** extra ([project.md](project.md)).

## Decided

### `DetectedFace`

Frozen dataclass (`eq=False`: it holds an array), one detected face. Every coordinate is **normalised to the image given**: `x` in `[0, 1]` left to right, `y` in `[0, 1]` top to bottom, whatever its resolution — the caller scales to pixels.

| Field | Type | Meaning |
|---|---|---|
| `box` | `tuple[float, float, float, float]` | `(x0, y0, x1, y1)`, unpadded, clamped to `[0, 1]` — [hand.md](hand.md)'s `DetectedHand.box` convention |
| `score` | `float \| None` | the backend's per-face detection confidence; `None` when it exposes none (MediaPipe's landmarker result carries no per-face score) |
| `nose` | `tuple[float, float] \| None` | the tip of the nose — the point a robot's head aims at |
| `eyes` | `tuple[tuple[float, float], tuple[float, float]] \| None` | the centres of the eyes, **the subject's right eye first, then the left** — in an unmirrored camera image the right eye is the one on the image's left |
| `orientation` | `tuple[float, float, float] \| None` | `(roll, pitch, yaw)` in radians, the head's rotation as seen from the camera, in the convention below; `None` from a backend that fits no head |
| `landmarks` | `np.ndarray \| None` | the backend's full landmark set, shape `(N, 3)`, `float32`, `x` / `y` normalised as above, `z` the backend's relative depth; `None` from a backend without one. For MediaPipe, the 478-point mesh (irises included) in MediaPipe's index order |

**The orientation convention**, stated in image terms so it holds for any backend:

- **`roll`** is positive when the face's eye line turns **clockwise** as seen in the image. It is the sign `atan2(left_y − right_y, left_x − right_x)` gives on `eyes` in image coordinates (y down): an upright face reads `0`, and a backend's fitted roll agrees in sign and — within a few degrees — in value with its own eye line.
- **`pitch`** is positive when the face tilts **down** (chin toward the chest).
- **`yaw`** is positive when the face turns toward the **image's right** (in an unmirrored image: the subject turning to their own left).
- A frontal, upright face facing the camera reads `(0, 0, 0)`.

`nose`, `eyes` and `orientation` are the three things a gaze consumer needs, named so it never indexes into a backend's landmark layout; `landmarks` is for consumers that do (alignment for a recogniser, blendshape-free expression cues, a mesh overlay).

### `FaceDetector` — the seam

```
class FaceDetector(Protocol):
    def detect(self, image_bgr: np.ndarray, ts: float) -> tuple[DetectedFace, ...]: ...
    def close(self) -> None: ...
```

- `image_bgr` is an `(H, W, 3)` `uint8` BGR array, **read-only** to the detector (it is a shared frame, [stream.md](stream.md) ownership rules): a backend that needs another layout converts into its own buffer.
- `ts` is the frame's time in seconds on a monotonic clock — a `Frame`'s `ts`, or any caller's own capture stamp (e.g. `time.monotonic()` at capture). A tracking backend derives its own video timestamps from it (below); a stateless one ignores it.
- Faces come back in any order; the caller selects.
- **Threading:** a detector is not thread-safe. It is called by **one caller at a time**, never concurrently; successive calls may come from different threads (a thread pool's worker per call). The constructor may run on yet another thread. `close()` is called once, after the last `detect`, from any thread. This is stricter than nothing and looser than "one fixed thread": it is what a stage's worker thread satisfies, and what a caller dispatching each frame to an asyncio thread pool satisfies too.
- `detect` after `close()` raises `RuntimeError`; `close()` twice is a no-op.

### `MediaPipeFaceDetector` — the shipped backend (Tasks API)

```
MediaPipeFaceDetector(max_faces: int = 1,
                      min_face_detection_confidence: float = 0.5,
                      min_face_presence_confidence: float = 0.5,
                      min_tracking_confidence: float = 0.5,
                      model_path: str | Path | None = None)
```

- Built on the **MediaPipe Tasks API** `mediapipe.tasks.python.vision.FaceLandmarker` — BlazeFace short-range detection, then the 478-point mesh, then a fitted head transform — with `output_facial_transformation_matrixes=True` and `output_face_blendshapes=False` (open question 4). Not the legacy `mp.solutions.face_mesh`.
- **`max_faces`** is MediaPipe's `num_faces`, validated `>= 1` (`ValueError`, raised before anything is imported or downloaded). Its default of one is MediaPipe's; a caller that wants everyone in view raises it. The trade [hand.md](hand.md) documents holds here too: in `VIDEO` mode detection re-runs only while fewer than `max_faces` faces are tracked, and at most `max_faces` come back, so with `max_faces=1` the first face locked onto is the only one reported.
- **Running mode `VIDEO`**, fed `ts` converted to integer milliseconds, forced strictly increasing (a `ts` at or before the previous one becomes the previous + 1 ms) — the same rule as `MediaPipeHandDetector`. Because the landmarker tracks, a detector instance belongs to **one video**: a caller feeding unrelated images builds one detector per image.
- **BGR → RGB** by a numpy channel swap (`np.ascontiguousarray(image_bgr[..., ::-1])`), wrapped in `mp.Image`; no `cv2` import ([project.md](project.md)).
- **`mediapipe` imported inside the constructor**, so `import vision_modules.face` succeeds with `numpy` alone ([project.md](project.md) "The core imports no optional package at module level"); the constructor without the extra raises `ImportError` naming `vision-modules[face-mediapipe]`.
- **The mapping** from one native face to a `DetectedFace`:
  - `box` — the extremes of the mesh's `x` / `y`, clamped to `[0, 1]`. A mesh box is tighter than a detector's (chin to brow, no hair): a consumer that sizes faces by box height reads a smaller face than from YuNet at the same distance.
  - `nose` — the nose-tip landmark; `eyes` — the iris centres, the subject's right then left. The indices are module constants pinned in code (MediaPipe's documented mesh: nose tip `1`, iris centres `468` / `473`), and the live test checks their sides on a frontal face (the right eye's `x` below the left eye's).
  - `orientation` — decomposed from the face's facial transformation matrix into `(roll, pitch, yaw)` in the convention above. The decomposition (axis order, MediaPipe's canonical face-model axes to the image's) is the implementation's to get right, and the convention is pinned by tests both ways: synthetic matrices in `tests/`, and in `tests-e2e/` real photos whose pose is known — the eye line and a rotated copy for `roll`, a face turned toward the image's right for the absolute sign of `yaw`, a face tilted down for the absolute sign of `pitch`.
  - `landmarks` — the mesh as a `(478, 3)` `float32` array.
  - `score` — `None` (the Tasks result has no per-face score).
- **Model bundle** `face_landmarker.task` (a few MB): not in the wheel; downloaded on first use from Google's model storage into the cache directory [hand.md](hand.md) defines (`~/.cache/vision-modules/`, `VISION_MODULES_CACHE` overrides), `model_path=` bypasses it. The URL is pinned in code (`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` at the time of writing). The download happens in the constructor, which is why a caller builds the detector off any latency-sensitive thread.
- `close()` closes the landmarker.

### Public names

`DetectedFace`, `FaceDetector` and `MediaPipeFaceDetector` live in `vision_modules.face` — the import path consumers rely on — and are re-exported from the package root like every other public name ([project.md](project.md)). Importing either costs a numpy-only user nothing.

### The `face-mediapipe` extra

`face-mediapipe = ["mediapipe>=0.10.35,<1.0"]` — the same MediaPipe range as `hand`. The face family's extras are named per backend (`face-<backend>`) because the family will hold backends with unrelated dependencies — a landmarker on MediaPipe today, a recogniser on OpenCV Zoo's SFace or on torch later (open question 2) — and a consumer installs only the one it runs. MediaPipe brings `opencv-contrib-python`, which replaces a plain `opencv-python` in the same environment ([project.md](project.md) already notes this for `hand`). The `dev` group installs it with every other extra.

### Consumers

- **Host applications** install `vision-modules[face-mediapipe]` and call `MediaPipeFaceDetector` directly from their own detection loop — typically dispatching each frame to a thread pool (`asyncio.to_thread`), one call at a time. The public contract they rely on is exactly: the import path `vision_modules.face`; the names `MediaPipeFaceDetector`, `DetectedFace`, `FaceDetector`; the `max_faces` keyword; `box`, `nose`, `eyes`, `orientation` in the conventions above; `ts` in seconds; the threading rule above; `close()`. Converting the normalised coordinates to pixels is theirs. A change to any of these is a breaking change.
- **A future `FaceStage`** in this library (open question 1) would hold a `FaceDetector` as [hand.md](hand.md)'s `HandStage` holds a `HandDetector` — owned or borrowed ([pipeline.md](pipeline.md)).

### Tests

- **`tests/test_face.py`** — no model, no network, no `mediapipe` needed: the landmarker is built behind a private module-level seam that tests replace with a fake returning scripted native results (a mesh and a transformation matrix per face). Asserts: the box is the mesh extremes, clamped; `nose` / `eyes` taken from the pinned indices, right eye first; `landmarks` shape and dtype; `score is None`; the orientation of synthetic transformation matrices — identity → `(0, 0, 0)`, a pure roll / pitch / yaw of ±15° → that angle alone with the sign the convention says; the video timestamps strictly increasing for equal and decreasing `ts`; the image handed to the fake is RGB, contiguous, and the caller's array untouched; `max_faces < 1` → `ValueError`; `detect` after `close()` → `RuntimeError`, `close()` twice fine. `tests/test_core_imports.py` covers the numpy-only side: with `mediapipe` blocked, `vision_modules.face` imports and `MediaPipeFaceDetector()` raises an `ImportError` naming `vision-modules[face-mediapipe]`.
- **`tests-e2e/test_face_live.py`** — the real model on **public-domain photos of known pose** in `tests-e2e/fixtures/`, each file's source and licence recorded in the fixtures README. Each test skips when its fixture is absent. A fresh detector per independent image (tracking, above).
  - `face_frontal.jpg` — one person, upright, facing the camera. With `max_faces=1`: one face found; its box contains its `nose` and both `eyes`; the right eye's `x` below the left eye's; `|pitch|` and `|yaw|` under 10°; the fitted roll within 5° of the eye-line roll. The photo rotated **clockwise** by `θ = ±15°` (canvas kept, face centred): fitted and eye-line roll both within 5° of the unrotated roll `+ θ`, sign included. Two copies side by side with `max_faces=2`: two faces with disjoint boxes. Ten sequential calls, each from a new thread joined before the next starts: all succeed and find the face — the threading rule checked, not assumed.
  - `face_yaw_right.jpg` — one person turned clearly (≈ 20° or more) toward the **image's right**: `yaw > 10°`; the same photo mirrored left-right: `yaw < −10°`.
  - `face_pitch_down.jpg` — one person facing the camera with the head tilted clearly **down**: `pitch > 10°`.

## Open questions

1. **`FaceStage`.** A shared stage mirroring [hand.md](hand.md) — `max_faces`, selection by area with hysteresis, slot-stable order, padded crops copied out of the frame, a `FaceResult` — over any `Upstream[FrameLike]`, holding a `FaceDetector`. Deferred until a consumer inside the library needs crops (a recogniser, an expression module); a host running the detector in its own loop needs no stage. When it lands, a stable `track_id` (nearest-centre matching across frames is enough to start) is worth having from the start rather than as hand.md's open question 3.
2. **A recogniser.** A `FaceRecognizer` module embedding face crops and matching them against a gallery (a folder of reference images per person), over any upstream whose items carry `frame_id`, `ts`, an image and faces with a box and a `track_id` — a host's own face report through a small adapter, or a `FaceStage`. Backends behind one seam: OpenCV Zoo's SFace (Apache-2.0, designed for YuNet's five landmarks) first, `facenet-pytorch` (MIT) if a few people are not separated reliably; each its own `face-<backend>` extra. Its own spec.
3. **Other detector backends.** YuNet through OpenCV (`cv2.FaceDetectorYN`, MIT weights) behind the same seam, for a consumer without MediaPipe: five landmarks, no fitted orientation (`orientation=None`), a `score`. MediaPipe's standalone `FaceDetector` (BlazeFace alone) gives six keypoints and no head fit, so it adds nothing the landmarker does not; not planned.
4. **Blendshapes.** The landmarker's 52 blendshapes (jaw open, lip movements) would feed a *who is speaking* cue combined with a microphone array's direction of arrival. An optional `blendshapes` field and an `output_face_blendshapes` switch; deferred until that module is specced.
5. **Mirrored input.** With [stream.md](stream.md)'s `mirror` option on, "the subject's right eye" lands on the image's right and the yaw sign flips relative to the physical head. The fields stay defined on the image given; a consumer feeding mirrored frames accounts for it. Revisit if a mirrored consumer appears.
