"""Record a run to runs/<id>/ — summary.json is the healer's input format."""

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path


class RunRecorder:
    def __init__(self, runs_dir: str | Path, label: str = ""):
        self.runs_dir = Path(runs_dir)
        self.label = label
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}-{secrets.token_hex(3)}"
        self._events: list[dict] = []

    def record_event(self, event: dict) -> None:
        self._events.append(event)

    def record_frame(self, turn: int, png: bytes) -> None:
        """Frames are written immediately — they're too big to buffer."""
        frames_dir = self.runs_dir / self.run_id / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        self._frame_seq = getattr(self, "_frame_seq", 0) + 1
        (frames_dir / f"{self._frame_seq:04d}-t{turn}.png").write_bytes(png)

    def finalize(self, fitness: dict, params: dict) -> Path:
        out = self.runs_dir / self.run_id
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "events.jsonl", "w") as f:
            for event in self._events:
                f.write(json.dumps(event) + "\n")
        (out / "summary.json").write_text(json.dumps({"run_id": self.run_id, "fitness": fitness, "params": params}))
        (out / "meta.json").write_text(
            json.dumps({"label": self.label, "recorded_at": datetime.now(timezone.utc).isoformat()})
        )
        return out
