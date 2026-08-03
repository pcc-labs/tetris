import json
from datetime import datetime

from tetris_agent.board import Features
from tetris_agent.events import SCHEMA, EventCollector, build_spawn_event
from tetris_agent.policy import Placement
from tetris_agent.publisher import JSONLPublisher, NoopPublisher


def test_envelope_has_exact_keys_and_utc_timestamp():
    event = build_spawn_event(3, "J", "O")
    assert set(event) == {"schema", "event_type", "turn", "occurred_at", "data"}
    assert event["schema"] == SCHEMA == "tetris.game.v1"
    assert event["event_type"] == "piece_spawn"
    assert event["turn"] == 3
    ts = datetime.fromisoformat(event["occurred_at"])
    assert ts.utcoffset().total_seconds() == 0


def test_collector_increments_turn_on_spawn_and_publishes():
    seen = []

    class Capture:
        def publish(self, event):
            seen.append(event)

    collector = EventCollector(Capture())
    collector.session("start", {"genome": {}})
    collector.spawn("J", "O")
    collector.decision(Placement(rotation=1, col=3, score=0.5))
    collector.locked(1, Features(holes=0, agg_height=4, bumpiness=2, max_height=2), misexec=0)
    collector.spawn("O", "T")
    assert collector.turn == 2
    types = [e["event_type"] for e in seen]
    assert types == ["session", "piece_spawn", "placement_decision", "piece_locked", "piece_spawn"]


def test_jsonl_publisher_writes_date_partitioned_lines(tmp_path):
    pub = JSONLPublisher(tmp_path, session_id="abc123")
    pub.publish(build_spawn_event(1, "I", "T"))
    pub.publish(build_spawn_event(2, "T", "S"))
    day_dirs = list(tmp_path.iterdir())
    assert len(day_dirs) == 1
    f = day_dirs[0] / "events-abc123.jsonl"
    lines = [json.loads(line) for line in f.read_text().splitlines()]
    assert [e["turn"] for e in lines] == [1, 2]


def test_noop_publisher_swallows():
    NoopPublisher().publish({"anything": 1})
