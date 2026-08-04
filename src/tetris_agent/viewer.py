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

    @app.websocket("/ws/produce")
    async def ws_produce(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                message = await ws.receive_text()
                await hub.broadcast(message)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
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
