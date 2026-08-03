"""Telemetry sinks. The agent never talks to a broker: JSONL on disk is the
integration point — a bridge process (see pokemon-kafka's game-event-bridge)
can tail these files into Kafka without the agent knowing."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class Publisher(Protocol):
    def publish(self, event: dict) -> None: ...


class NoopPublisher:
    def publish(self, event: dict) -> None:
        return None


class JSONLPublisher:
    def __init__(self, root: str | Path, session_id: str):
        self.root = Path(root)
        self.session_id = session_id

    def publish(self, event: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.root / day / f"events-{self.session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")
