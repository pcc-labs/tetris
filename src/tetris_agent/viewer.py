"""Demo viewer: a Game Boy in the browser.

Serves the static shell UI, a replay API over runs/, and a live hub —
the agent pushes JSON to /ws/produce, every browser on /ws/live sees it.
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class LiveHub:
    """Fan-out: producer messages are rebroadcast to every subscriber."""

    def __init__(self):
        self.subscribers: set[WebSocket] = set()
        # The most recent session-start, replayed to late-joining tabs so a
        # browser opened mid-arm still learns who is playing (a benchmark arm
        # can run for many minutes). Cleared on session end.
        self.last_session_start: str | None = None

    def note(self, message: str) -> None:
        """Remember session boundaries. Cheap: frames are base64 payloads that
        cannot contain a quoted "session", so they skip the JSON parse."""
        if '"session"' not in message:
            return
        try:
            parsed = json.loads(message)
        except ValueError:
            return
        event = parsed.get("event") or {}
        if event.get("event_type") != "session":
            return
        phase = (event.get("data") or {}).get("phase")
        if phase == "start":
            self.last_session_start = message
        elif phase == "end":
            self.last_session_start = None

    async def broadcast(self, message: str) -> None:
        dead = []
        for ws in self.subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subscribers.discard(ws)


def _run_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
        meta = {}
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        return {
            "run_id": summary.get("run_id", run_dir.name),
            "label": meta.get("label", ""),
            "recorded_at": meta.get("recorded_at", ""),
            "fitness": summary.get("fitness", {}),
            "params": summary.get("params", {}),
        }
    except json.JSONDecodeError:
        logger.warning("unreadable summary in %s", run_dir)
        return None


def create_app(runs_dir: str | Path = "runs") -> FastAPI:
    runs_dir = Path(runs_dir)
    app = FastAPI(title="tetris-agent viewer")
    hub = LiveHub()
    app.state.hub = hub
    app.state.input_peers = set()

    @app.middleware("http")
    async def no_stale_frontend(request, call_next):
        # A cached app.js from before a code change silently breaks browser
        # play (keys go nowhere). Dev tool: every tab revalidates, 304s are cheap.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["cache-control"] = "no-cache"
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/runs")
    async def list_runs():
        runs = []
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                summary = _run_summary(run_dir)
                if summary:
                    frames_dir = run_dir / "frames"
                    summary["frame_count"] = len(list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
                    runs.append(summary)
        return runs

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        run_dir = runs_dir / run_id
        summary = _run_summary(run_dir)
        if summary is None:
            raise HTTPException(status_code=404, detail="run not found")
        events = []
        events_path = run_dir / "events.jsonl"
        if events_path.exists():
            events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
        frames_dir = run_dir / "frames"
        frame_names = sorted(f.name for f in frames_dir.glob("*.png")) if frames_dir.exists() else []
        summary["events"] = events
        summary["frames"] = [f"/api/runs/{run_id}/frames/{name}" for name in frame_names]
        return summary

    @app.get("/api/runs/{run_id}/frames/{name}")
    async def run_frame(run_id: str, name: str):
        path = (runs_dir / run_id / "frames" / name).resolve()
        if not path.is_file() or runs_dir.resolve() not in path.parents:
            raise HTTPException(status_code=404, detail="frame not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/benchmarks")
    async def list_benchmarks():
        """Newest benchmark run first: its ranked summary rows."""
        bench_dir = Path("data/benchmarks")
        out = []
        if bench_dir.exists():
            for path in sorted(bench_dir.glob("benchmark-*.json"), reverse=True):
                try:
                    saved = json.loads(path.read_text())
                except json.JSONDecodeError:
                    logger.warning("unreadable benchmark %s", path)
                    continue
                out.append(
                    {
                        "id": path.stem,
                        "recorded_at": saved.get("recorded_at", ""),
                        "summary": saved.get("summary", []),
                    }
                )
        return out

    @app.websocket("/ws/produce")
    async def ws_produce(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                message = await ws.receive_text()
                hub.note(message)
                await hub.broadcast(message)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/input")
    async def ws_input(ws: WebSocket):
        # Echo hub for browser play: the browser writes keystrokes, the game
        # process reads them. Every peer hears everyone but itself, so the
        # endpoint doesn't care which side is which.
        await ws.accept()
        peers: set[WebSocket] = app.state.input_peers
        peers.add(ws)
        try:
            while True:
                message = await ws.receive_text()
                dead = []
                for peer in peers:
                    if peer is ws:
                        continue
                    try:
                        await peer.send_text(message)
                    except Exception:
                        dead.append(peer)
                for peer in dead:
                    peers.discard(peer)
        except WebSocketDisconnect:
            pass
        finally:
            peers.discard(ws)

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        # Replay before subscribing, so the identity always precedes any
        # broadcast this tab sees.
        if hub.last_session_start is not None:
            try:
                await ws.send_text(hub.last_session_start)
            except Exception:
                pass
        hub.subscribers.add(ws)
        try:
            while True:
                # Keep the connection open; subscribers only listen.
                await asyncio.sleep(3600)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            hub.subscribers.discard(ws)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main(argv=None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="tetris-viewer")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    uvicorn.run(create_app(args.runs_dir), host=args.host, port=args.port)
    return 0
