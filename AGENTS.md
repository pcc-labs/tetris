# Agent notes

- Use `uv` for everything: `uv sync`, `uv run pytest`, `uv run ruff check .`.
- Default `pytest` run excludes emulator tests (`-m "not rom"` via addopts). Run everything with `uv run pytest -m ""`.
- The ROM lives at `rom/tetris.gb` and is gitignored — never commit it.
- Mutable runtime state (genome, healer state, telemetry) lives under `data/`; recorded runs under `runs/`. Both gitignored.
- One installable package (`src/tetris_agent/`), absolute imports only. No subprocess calls *between components*. Three carve-outs, all of them a component shelling out to an external tool rather than to another component: discovery → `claude`, `pi_policy` → `pi`, and `power` → `powermetrics`/`nvidia-smi` (Linux RAPL and hwmon are plain file reads and need no exemption).
