# Vision Modules

Independent vision modules that run over a video stream and turn it into perception a
multimodal robot, system, or agent can act on. Each module solves one perceptual job behind
a uniform interface, so you enable only the modules your use case needs.

Status: early scaffold — no modules built yet.

## Development

This project is built spec-first. Read [AGENTS.md](AGENTS.md), then
[specs/_index.md](specs/_index.md) and [plans/_index.md](plans/_index.md).

```
uv sync --dev
uv run ruff check .
uv run pyright
uv run pytest
```
