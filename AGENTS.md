# Agent notes

- Use `uv` for everything: `uv sync`, `uv run pytest`, `uv run ruff check .`.
- Default `pytest` run excludes emulator tests (`-m "not rom"` via addopts). Run everything with `uv run pytest -m ""`.
- The ROM lives at `rom/tetris.gb` and is gitignored — never commit it.
- Mutable runtime state (genome, healer state, telemetry) lives under `data/`; recorded runs under `runs/`. Both gitignored.
- One installable package (`src/tetris_agent/`), absolute imports only. No subprocess calls between components except discovery → `claude`.
