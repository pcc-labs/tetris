"""LiveStreamer URL handling."""

from tetris_agent.live import LiveStreamer


def test_http_viewer_url_becomes_a_websocket_url():
    # Callers hold the viewer's http:// address; the producer socket must not
    # silently fail to connect because of the scheme (streaming degrades
    # quietly by design, which turned this exact mistake invisible).
    assert LiveStreamer("http://127.0.0.1:8000").url == "ws://127.0.0.1:8000/ws/produce"
    assert LiveStreamer("https://viewer.example").url == "wss://viewer.example/ws/produce"
    assert LiveStreamer("ws://127.0.0.1:8000").url == "ws://127.0.0.1:8000/ws/produce"
