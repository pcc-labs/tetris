"""Best-effort WebSocket producer for the viewer's live hub.

Same paradigm as pokemon-kafka's LiveProducer: connect lazily, drop and reset
on any failure, never block or crash gameplay.
"""

import base64
import json
import logging

logger = logging.getLogger(__name__)

try:
    from websockets.sync.client import connect
except Exception:  # pragma: no cover - optional dependency guard
    connect = None


class LiveStreamer:
    def __init__(self, viewer_url: str):
        self.url = viewer_url.rstrip("/") + "/ws/produce"
        self._ws = None
        self._warned = False

    def _ensure(self):
        if self._ws is None and connect is not None:
            self._ws = connect(self.url, open_timeout=2)

    def send(self, message: dict) -> None:
        try:
            self._ensure()
            if self._ws is not None:
                self._ws.send(json.dumps(message))
        except Exception:
            if not self._warned:
                logger.info("viewer not reachable at %s; streaming disabled until it is", self.url)
                self._warned = True
            self._ws = None

    def send_frame(self, turn: int, png: bytes) -> None:
        self.send({"type": "frame", "turn": turn, "png": base64.b64encode(png).decode()})

    def send_event(self, event: dict) -> None:
        self.send({"type": "event", "event": event})

    def close(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        self._ws = None


class LiveEventSink:
    """Publisher-shaped adapter so the collector can tee events to the stream."""

    def __init__(self, streamer: LiveStreamer):
        self.streamer = streamer

    def publish(self, event: dict) -> None:
        self.streamer.send_event(event)
