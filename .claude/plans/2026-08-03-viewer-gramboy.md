# Viewer Phase: Game Boy Browser UI + Session Replay

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Continues the 2026-08-03-tetris-self-healing-agent plan.

**Goal:** Mirror pokemon-kafka's demo experience — but the browser shows a Game Boy (DMG-01) shell rendering the game, with live streaming and replay of recorded runs.

**Architecture:** The agent captures PNG frames (spawn + lock) into `runs/<id>/frames/` and, with `--live`, streams throttled frames+events over a best-effort WebSocket to the viewer. The viewer is a FastAPI app (`tetris-viewer`) serving a static Game Boy UI, a runs/replay API over `runs/`, and a live broadcast hub (`/ws/produce` → `/ws/live`). Replay steps through stored frames with HUD state derived from events.

**Tech Stack:** fastapi, uvicorn, websockets, pillow (new deps); httpx (dev, for TestClient).

## Tasks

### Task 16: Frame capture
- `Emulator.screenshot() -> bytes` (PNG via `pyboy.screen.image`).
- `RunRecorder.record_frame(turn: int, png: bytes) -> None` writes `frames/<seq:04d>-t<turn>.png` immediately (not buffered).
- Agent: capture at spawn and after lock when recorder present; also feed live streamer.
- Tests: rom test for PNG magic; pure test for frame file naming.

### Task 17: Viewer server (`src/tetris_agent/viewer.py`)
- `create_app(runs_dir) -> FastAPI`:
  - `GET /` → static/index.html; `/static/*` assets.
  - `GET /api/runs` → newest-first `[{run_id, label, fitness}]` from summary/meta.
  - `GET /api/runs/{id}` → `{run_id, label, fitness, params, events, frames: [urls]}`.
  - `GET /api/runs/{id}/frames/{name}` → PNG file.
  - `WS /ws/produce` (agent pushes JSON) → rebroadcast to all `WS /ws/live` subscribers.
- `tetris-viewer` entrypoint: uvicorn 127.0.0.1:8000, `--runs-dir`, `--port`.
- Tests: TestClient for runs list/detail/frame; 404s.

### Task 18: Game Boy UI (`src/tetris_agent/static/`)
- DMG-01 shell in CSS: gray case, screen bezel with "DOT MATRIX WITH STEREO SOUND", olive-green LCD, d-pad, A/B, START/SELECT. Canvas (160×144) inside the LCD, pixelated upscale.
- HUD beside shell: score/lines/level, piece count, genome badge, event ticker.
- Modes: LIVE (subscribe /ws/live) and REPLAY (run selector, play/pause, speed 1×/2×/4×, scrubber over frames; HUD replays event stream in step).

### Task 19: Live streaming
- `LiveStreamer` (best-effort, pokemon LiveProducer paradigm): connect lazily to `ws://…/ws/produce`, `send(dict)`, drop-and-reset on failure, never block gameplay.
- Throttled mid-drop frames via an `Emulator` frame hook (every ~12 ticks when live).
- CLI: `--live` now = stream to viewer (speed=1, still headless unless `--window SDL2`); `--viewer-url` default `ws://127.0.0.1:8000`.
- README demo section; commit + push.
