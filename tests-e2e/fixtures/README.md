# Live-tier fixtures

`hand_<gesture>.jpg` — a real photo with one clearly visible hand making
`<gesture>`, where `<gesture>` is one of the HaGRID class names
(`specs/gesture_classifier.md`, e.g. `hand_palm.jpg`, `hand_stop.jpg`).

Used by:
- `tests-e2e/test_hand_live.py` and `tests-e2e/test_hand_pipeline_live.py` as
  the default hand-detection input (any one file; gesture doesn't matter
  there) when `VISION_MODULES_HAND_IMAGE` isn't set.
- `tests-e2e/test_hand_pipeline_live.py`'s classification test, parametrized
  over every `hand_<gesture>.jpg` found here, asserting the full pipeline
  actually predicts the photographed gesture — the filename is the ground
  truth label.

`no_hand.jpg` — a real photo with no hand in frame. Used by
`tests-e2e/test_hand_pipeline_live.py`'s negative-path test, asserting the
full pipeline reports `present=False` end to end instead of a stale or
fabricated result.

`two_hands.jpg` (optional) — a real photo with two clearly visible hands of
different apparent size. Used by `tests-e2e/test_hand_live.py` to check that the
detector returns two hands when asked for two and one when asked for one; the
test skips when the file is absent.

The hand photos above are original photos contributed to the project (not
sourced from a third-party dataset), so they carry no external
license/attribution obligations — covered by this repo's own license like any
other file here.

## Face fixtures

Public-domain NASA photos (US federal government works) of a known head pose, used by
`tests-e2e/test_face_live.py` to pin the sign of `DetectedFace.orientation`
(`specs/face.md`) against the real model. Each is a crop of a downscaled
Wikimedia Commons thumbnail (≤ 640 px on the long side); the pose was checked by
eye, not taken from the detector.

| File | Pose | Source | Author | Licence |
|---|---|---|---|---|
| `face_frontal.jpg` | upright, facing the camera | [Kalpana Chawla, NASA photo portrait in orange suit](https://commons.wikimedia.org/wiki/File:Kalpana_Chawla,_NASA_photo_portrait_in_orange_suit.jpg) (NASA JSC2002e2532) | NASA | Public domain (US federal government work) |
| `face_yaw_right.jpg` | head turned toward the image's right | [Profiles in Leadership: Joe Schmid is loving life as a NASA flight surgeon and Air Force Reservist](https://commons.wikimedia.org/wiki/File:Profiles_in_Leadership-_Joe_Schmid_is_loving_life_as_a_NASA_flight_surgeon_and_Air_Force_Reservist_150519-F-GS664-033.jpg) (150519-F-GS664-033) | Bill Ingalls, NASA | Public domain (US federal government work) |
| `face_pitch_down.jpg` | head tilted down, looking at the hands | [View of STS-106 MS cosmonaut Malenchenko suiting up for an EVA](https://commons.wikimedia.org/wiki/File:View_of_STS-106_MS_cosmonaut_Malenchenko_suiting_up_for_an_EVA.jpg) (NASA jsc2000e23502) | NASA | Public domain (US federal government work) |
