import json

from fastapi.testclient import TestClient

from tetris_agent.viewer import create_app


def make_run(runs_dir, run_id="20260803-120000-abc123", label="demo", frames=2):
    run_dir = runs_dir / run_id
    (run_dir / "frames").mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"run_id": run_id, "fitness": {"score": 400, "lines": 10}, "params": {"w_holes": -0.35}})
    )
    (run_dir / "meta.json").write_text(json.dumps({"label": label, "recorded_at": "2026-08-03T12:00:00+00:00"}))
    (run_dir / "events.jsonl").write_text(
        '{"event_type": "piece_spawn", "turn": 1}\n{"event_type": "piece_locked", "turn": 1}\n'
    )
    for i in range(1, frames + 1):
        (run_dir / "frames" / f"{i:04d}-t{i}.png").write_bytes(b"\x89PNG fake")
    return run_id


def test_list_runs_newest_first(tmp_path):
    make_run(tmp_path, "20260803-100000-aaaaaa", label="old")
    make_run(tmp_path, "20260803-120000-bbbbbb", label="new")
    client = TestClient(create_app(tmp_path))
    runs = client.get("/api/runs").json()
    assert [r["label"] for r in runs] == ["new", "old"]
    assert runs[0]["frame_count"] == 2
    assert runs[0]["fitness"]["score"] == 400


def test_run_detail_includes_events_and_frame_urls(tmp_path):
    run_id = make_run(tmp_path)
    client = TestClient(create_app(tmp_path))
    detail = client.get(f"/api/runs/{run_id}").json()
    assert len(detail["events"]) == 2
    assert detail["frames"] == [
        f"/api/runs/{run_id}/frames/0001-t1.png",
        f"/api/runs/{run_id}/frames/0002-t2.png",
    ]
    png = client.get(detail["frames"][0])
    assert png.status_code == 200
    assert png.content == b"\x89PNG fake"


def test_missing_run_and_frame_404(tmp_path):
    run_id = make_run(tmp_path)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get(f"/api/runs/{run_id}/frames/../../summary.json").status_code == 404
    assert client.get(f"/api/runs/{run_id}/frames/nope.png").status_code == 404


def test_live_hub_rebroadcasts_producer_messages(tmp_path):
    client = TestClient(create_app(tmp_path))
    with client.websocket_connect("/ws/live") as live:
        with client.websocket_connect("/ws/produce") as producer:
            producer.send_text('{"type": "frame", "turn": 1}')
            message = live.receive_text()
    assert json.loads(message) == {"type": "frame", "turn": 1}


def test_index_serves_gameboy_shell(tmp_path):
    client = TestClient(create_app(tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]


def test_live_streamer_survives_unreachable_viewer():
    from tetris_agent.live import LiveStreamer

    streamer = LiveStreamer("ws://127.0.0.1:1")  # nothing listens here
    streamer.send_frame(1, b"\x89PNG fake")
    streamer.send_event({"event_type": "piece_spawn"})
    streamer.close()  # no exception = pass


def test_benchmarks_endpoint_returns_newest_first(tmp_path, monkeypatch):
    bench = tmp_path / "data" / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "benchmark-20260803-100000.json").write_text(
        json.dumps({"recorded_at": "2026-08-03T10:00:00+00:00", "summary": [{"arm": "old"}]})
    )
    (bench / "benchmark-20260803-120000.json").write_text(
        json.dumps({"recorded_at": "2026-08-03T12:00:00+00:00", "summary": [{"arm": "new"}]})
    )
    (bench / "benchmark-20260803-130000.json").write_text("{broken")
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(tmp_path / "runs"))
    runs = client.get("/api/benchmarks").json()
    assert [r["summary"][0]["arm"] for r in runs] == ["new", "old"]  # corrupt file skipped


def test_benchmarks_endpoint_empty_without_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/benchmarks").json() == []


def test_input_hub_echoes_to_other_peers_not_sender(tmp_path):
    client = TestClient(create_app(tmp_path))
    with client.websocket_connect("/ws/input") as game:
        with client.websocket_connect("/ws/input") as browser:
            browser.send_text('{"type": "input", "button": "left", "action": "press"}')
            message = json.loads(game.receive_text())
            assert message == {"type": "input", "button": "left", "action": "press"}
            # The sender must not hear its own keystrokes back.
            game.send_text('{"type": "input", "button": "a", "action": "press"}')
            echoed = json.loads(browser.receive_text())
            assert echoed["button"] == "a"


def test_static_and_index_are_served_no_cache(tmp_path):
    # A stale cached app.js silently breaks browser play; the viewer is a dev
    # tool, so make every tab revalidate.
    client = TestClient(create_app(tmp_path))
    assert client.get("/").headers.get("cache-control") == "no-cache"
    assert client.get("/static/app.js").headers.get("cache-control") == "no-cache"


def test_late_subscriber_receives_the_last_session_start():
    """A tab opened mid-arm must still learn who is playing: the hub replays
    the most recent session-start event to every new /ws/live subscriber."""
    client = TestClient(create_app("runs"))
    start = {
        "type": "event",
        "event": {"event_type": "session", "data": {"phase": "start", "model": "pi/gpt-oss:20b"}},
    }
    with client.websocket_connect("/ws/produce") as producer:
        producer.send_text(json.dumps(start))
        with client.websocket_connect("/ws/live") as late:
            # A sentinel frame after the late join: without replay, this frame
            # (not the session start) would be the first message received.
            producer.send_text('{"type": "frame", "turn": 9}')
            assert json.loads(late.receive_text()) == start
            assert json.loads(late.receive_text()) == {"type": "frame", "turn": 9}


def test_session_end_clears_the_replay_so_idle_tabs_stay_idle():
    client = TestClient(create_app("runs"))
    start = {"type": "event", "event": {"event_type": "session", "data": {"phase": "start", "model": "m"}}}
    end = {"type": "event", "event": {"event_type": "session", "data": {"phase": "end"}}}
    with client.websocket_connect("/ws/produce") as producer:
        producer.send_text(json.dumps(start))
        producer.send_text(json.dumps(end))
        with client.websocket_connect("/ws/live") as late:
            with client.websocket_connect("/ws/produce") as p2:
                p2.send_text('{"type": "frame", "turn": 9}')
            assert json.loads(late.receive_text()) == {"type": "frame", "turn": 9}
