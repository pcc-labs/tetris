"""LiveStreamer URL handling."""

import json

from tetris_agent.live import LiveStreamer


def test_http_viewer_url_becomes_a_websocket_url():
    # Callers hold the viewer's http:// address; the producer socket must not
    # silently fail to connect because of the scheme (streaming degrades
    # quietly by design, which turned this exact mistake invisible).
    assert LiveStreamer("http://127.0.0.1:8000").url == "ws://127.0.0.1:8000/ws/produce"
    assert LiveStreamer("https://viewer.example").url == "wss://viewer.example/ws/produce"
    assert LiveStreamer("ws://127.0.0.1:8000").url == "ws://127.0.0.1:8000/ws/produce"


class _FakeWs:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)


def test_messages_carry_their_slot():
    # The viewer routes frames to screens by slot; a race runs one streamer per
    # lane, and slot 0 is the single-screen LIVE tab every older producer uses.
    solo, lane2 = LiveStreamer("ws://x"), LiveStreamer("ws://x", slot=2)
    solo._ws, lane2._ws = _FakeWs(), _FakeWs()

    solo.send_frame(1, b"png")
    lane2.send_event({"event_type": "session"})

    assert json.loads(solo._ws.sent[0])["slot"] == 0
    assert json.loads(lane2._ws.sent[0])["slot"] == 2
