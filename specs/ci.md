---
code:
  - .github/workflows/ci.yml
  - pyproject.toml
tests:
---

# Continuous integration

**Status:** Stable

## Purpose

Both test tiers run on GitHub's hosted runners for every pull request and every push to `main`. This makes the verification gate in [AGENTS.md](../AGENTS.md) (lint, type check, tests) a machine's verdict on each change, not only a local command. CI runs what the tiers already run without credentials or hardware ([testing.md](testing.md)): the whole fast tier, and the live tier over the committed fixture images, with each skip read as part of the verdict. Like [testing.md](testing.md), this is a cross-cutting practice, not a runtime concept. Nothing here ships in the library; the workflow is the only file that implements it.

## Decided

### The runners

- **GitHub-hosted runners on the free tier.** The static gate and the live tier run on `ubuntu-24.04` (x86_64). The fast tier runs there and also on `macos-15` (arm64) and `windows-2025` (x86_64), the three platforms the library supports ([project.md](project.md) "Platforms"). Each image is pinned by name rather than `*-latest`, so the OS under the jobs changes only when someone bumps the pin. There is no self-hosted runner. The repo is public, so every runner is free.
- **The fast tier runs on every supported platform.** It needs no network, camera or model ([testing.md](testing.md)), so it runs unchanged on all three runners. Each runner is where that platform's own behaviour meets a real machine: the dependency markers of the lock (CPU torch from PyTorch's index on Linux, PyPI's wheel elsewhere), file paths and atomic renames (`ensure_model`), thread and timer timing in the pipeline runtime, and the subprocess import test. Development happens on macOS, so the Linux and Windows entries are where those platforms are exercised. A failure that shows up only on one platform is a library bug to fix, not a reason to drop the platform.
- **The live tier runs on Linux only.** The live tests run the real models (MediaPipe hand and face landmarkers, the HaGRID ViT) over fixture images, so they could run anywhere. But a macOS runner would mostly repeat the developer's own local run, on worse hardware. GitHub's macOS runners also have no MPS device, so they would not exercise the one path that exists only on macOS (`select_device` → `mps`). Linux is the platform no local run covers, and its x86_64 MediaPipe build is the most likely place for the numeric tolerances (face orientation) to drift away from Apple Silicon. See open question 1.
- **One Python version**, 3.12: the floor of `requires-python` and the version `.python-version` names. uv installs it on the runner.
- **Only the live job has system packages.** Every Python dependency comes as a wheel on all three platforms. On Linux, though, MediaPipe's native library loads the system's OpenGL ES and EGL libraries, which the Ubuntu runner image lacks. The `e2e` job apt-installs `libgles2` and `libegl1` before the sync, the same recipe the README gives Linux users ([project.md](project.md) "Platforms"). The fast tier never instantiates a MediaPipe task, so it needs no apt step, and macOS and Windows need none at all.

### The workflow

There is one file, `.github/workflows/ci.yml`. It triggers on `pull_request`, on `push` to `main`, and on `workflow_dispatch`. Concurrency is one run per ref: a newer push cancels the older run still in flight.

| Job | What it runs |
|---|---|
| `check` | `uv sync --locked`, then `ruff check .`, `ruff format --check .` and `pyright`: the static gate, on Linux |
| `fast-tier` | `uv sync --locked`, then `pytest`, the fast tier, **once per platform**: a three-entry matrix (`ubuntu-24.04`, `macos-15`, `windows-2025`) with `fail-fast: false`, so one platform's failure leaves the others running |
| `e2e` | `libgles2` and `libegl1` apt-installed, `uv sync --locked`, the model cache restored, then `pytest tests-e2e -rs`: the live tier on Linux |

The five jobs run side by side, each on its own runner, and none waits for another. A run takes as long as its slowest job. Putting the tiers behind the static gate would only save runner time on a commit that fails lint, which costs nothing on a public repo, and it would add the gate's time to every green run. Each job installs its own environment.

- **`--locked`.** The sync fails when `uv.lock` doesn't match `pyproject.toml`, so a dependency edit can only land together with its relock.
- **The `dev` group installs whole**, as it does locally: every extra (`opencv`, `hand`, `face-mediapipe`) and the `demo` group ([project.md](project.md)). The fast tier imports the demo's helpers, and the live tier needs every model backend. Torch is the heaviest dependency. On Linux it resolves from the PyTorch CPU index ([project.md](project.md) "Runtime dependencies"), never with the CUDA libraries, so each entry installs the full environment in seconds rather than downloading gigabytes.
- **The format check uses the same command as a local run**: `ruff format --check .`, the check form of the `ruff format .` in [AGENTS.md](../AGENTS.md) "Commands".
- **The fast tier runs as it does locally**: plain `uv run pytest`, serial, with no extra flags and no platform list. The tier takes a few seconds, so it has nothing to parallelise.
- **The live tier runs as it does locally**, serial. `-rs` prints every skip into the job log.
- **Each job has a `timeout-minutes`** well under GitHub's default, so a hung thread fails the job in minutes rather than hours.

### What the live job provides, and what it leaves out

- **Network, and a model cache.** The live tier downloads its models on first use: the MediaPipe `.task` bundles from Google's storage into `~/.cache/vision-modules` (`default_model_path`, about 11 MB), and the HaGRID ViT from the Hugging Face Hub into `~/.cache/huggingface` (about 330 MB). The workflow caches both folders. The cache key is the hash of the modules that name the models (`hand.py`, `face.py`, `gesture_classifier.py`), with no prefix restore key. `ensure_model` trusts a file that already exists under its name, so a model URL or id edited in code has to start from an empty cache and never be served the old weights. A cache miss just downloads again. If the download fails, the job **fails**: the live tier never skips a model it cannot fetch ([testing.md](testing.md) "Live tier: skip without credentials").
- **Fixtures, not a camera.** Every model test runs on the photos committed under `tests-e2e/fixtures/`. The runner has no camera, so the webcam test stays behind `VISION_MODULES_CAMERA` and skips.
- **Expected skips.** A skip is not a pass. On the runner, the following tests skip by design, and any other skip in the job log is an environment regression to investigate:

| Skips on the runner | Why |
|---|---|
| `test_stream_live.py::test_webcam_publishes_bgr_frames` | no camera on a runner; `VISION_MODULES_CAMERA` is unset |
| `test_hand_live.py::test_mediapipe_detector_returns_at_most_num_hands` | `two_hands.jpg` is an optional fixture, not committed (`tests-e2e/fixtures/README.md`); the test runs once that photo is added |

- **Never in CI:** a webcam, a GPU (the gesture classifier's `cuda` / `mps` paths stay a local run), and the Gradio demo served for real (its helpers are covered by the fast tier).

### Secrets and protection

- **No secret is required.** No test in either tier needs a credential today. A future key-gated live test reads its key through `require_env` and skips when the repository holds no such secret, and always on a fork's pull request, which never receives one.
- **The status checks to require on `main`** are `check`, the three `fast-tier` entries and `e2e`. Requiring them is a repository setting on GitHub, outside the repo.

## Relationship to the other specs

- **[testing.md](testing.md):** the two tiers and the skip rules CI runs on. CI is the hosted run of that strategy and adds no rule of its own about what a test does.
- **[project.md](project.md):** the supported platforms the fast tier's matrix lists, the `dev` group CI installs whole, and the PyTorch CPU index for Linux that keeps the install small. `.github/workflows/` is part of the repo shape.
- **[AGENTS.md](../AGENTS.md):** the verification gate CI automates.

## Open questions

1. **The live tier on macOS and Windows.** Each would be one more matrix entry and free on a public repo. It is deferred until a model regression slips through on one of those platforms and not on Linux; the developer's own macOS runs cover macOS today.
2. **More than one Python version.** `requires-python` is `>=3.12` with no upper bound, and the lock resolves for 3.13 and 3.14 too. A Python axis on the fast tier would test that claim. It is deferred until a dependency or a language change makes the floor and the newest version behave differently.
