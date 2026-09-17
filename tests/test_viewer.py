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


# ── label deck ──
# A real run for the deck: summary.json for the shelf plus events the replay verifies.
def make_human_run(runs_dir, run_id="20260808-120000-abc123"):
    from test_traces import O_AT_0, THEN_O_AT_2, write_run

    run_dir = write_run(runs_dir, run_id, "human", [("O", "O", 0, 0, 0, O_AT_0), ("O", "I", 0, 2, 0, THEN_O_AT_2)])
    (run_dir / "summary.json").write_text(json.dumps({"run_id": run_id, "fitness": {"score": 1}, "params": {}}))
    return run_id


FAKE_ANSWERS = {
    "verdict": {
        "type": "choice",
        "choice": "promote",
        "probabilities": {"promote": 0.7, "neutral": 0.2, "exclude": 0.1},
        "confidence": 0.7,
    },
    "quality": {"type": "score", "score": 3.2, "legend": {}, "probabilities": {}, "confidence": 0.6},
    "creates_hole": {"type": "noul", "noul": 0.05},
}


def fake_judge(state, questions=None):
    fake_judge.states.append(state)
    return {"model": "jev-test", "answers": FAKE_ANSWERS, "usage": {"input_tokens": 500, "output_tokens": 0}}


fake_judge.states = []


def test_decisions_endpoint_grades_and_serves_each_turn(tmp_path):
    run_id = make_human_run(tmp_path)
    client = TestClient(create_app(tmp_path))
    body = client.get(f"/api/runs/{run_id}/decisions").json()
    decisions = body["decisions"]
    assert body["placed"] == 2
    assert [d["turn"] for d in decisions] == [1, 2]
    assert decisions[0]["piece"] == "O" and decisions[0]["label"] is None
    assert decisions[0]["grade"]["legal_count"] > 1
    assert client.get("/api/runs/nope/decisions").status_code == 404


def test_labels_round_trip_through_the_api_and_reach_the_miner(tmp_path):
    from tetris_agent.traces import mine_run

    run_id = make_human_run(tmp_path)
    client = TestClient(create_app(tmp_path))
    put = client.put(f"/api/runs/{run_id}/labels/2", json={"verdict": "exclude", "jev": FAKE_ANSWERS})
    assert put.status_code == 200 and put.json()["verdict"] == "exclude"
    assert client.get(f"/api/runs/{run_id}/decisions").json()["decisions"][1]["label"]["source"] == "human"
    assert [e.col for e in mine_run(tmp_path / run_id)] == [0]
    assert client.put(f"/api/runs/{run_id}/labels/2", json={"verdict": "maybe"}).status_code == 422
    assert client.put(f"/api/runs/{run_id}/labels/99", json={"verdict": "promote"}).status_code == 404
    assert client.delete(f"/api/runs/{run_id}/labels/2").json() == {"cleared": 2}
    assert client.delete(f"/api/runs/{run_id}/labels/2").status_code == 404
    assert [e.col for e in mine_run(tmp_path / run_id)] == [0, 2]


def test_judge_endpoint_builds_the_state_and_relays_the_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    run_id = make_human_run(tmp_path)
    fake_judge.states.clear()
    client = TestClient(create_app(tmp_path, judge=fake_judge))
    res = client.post(f"/api/runs/{run_id}/decisions/2/judge")
    assert res.status_code == 200
    assert res.json()["answers"]["verdict"]["choice"] == "promote"
    assert res.json()["model"] == "jev-test"
    [state] = fake_judge.states
    assert state["placement"] == {"rotation": 0, "col": 2}
    assert state["board_before"][16:] == ["##........", "##........"]
    assert "oracle" in state
    assert client.post(f"/api/runs/{run_id}/decisions/99/judge").status_code == 404


def test_judge_endpoint_reports_a_missing_key_as_setup_not_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    run_id = make_human_run(tmp_path)
    client = TestClient(create_app(tmp_path, judge=fake_judge))
    questions = client.get("/api/jev/questions").json()
    assert questions["setup"]["configured"] is False
    assert {q["id"] for q in questions["questions"]} >= {"verdict", "quality"}
    res = client.post(f"/api/runs/{run_id}/decisions/1/judge")
    assert res.status_code == 503
    assert res.json()["detail"]["reason"] == "unconfigured"


def test_judge_endpoint_surfaces_upstream_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    run_id = make_human_run(tmp_path)

    def broken(state, questions=None):
        raise RuntimeError("TypeSafe 529: overloaded")

    client = TestClient(create_app(tmp_path, judge=broken))
    res = client.post(f"/api/runs/{run_id}/decisions/1/judge")
    assert res.status_code == 502
    assert "529" in res.json()["detail"]["error"]
