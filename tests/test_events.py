import json
from datetime import datetime

from tetris_agent.board import Features
from tetris_agent.events import SCHEMA, EventCollector, build_decision_event, build_spawn_event
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


def test_decision_event_carries_no_timing_keys_by_default():
    event = build_decision_event(2, Placement(rotation=1, col=3, score=0.5))
    assert set(event["data"]) == {"rotation", "col", "score"}


def test_decision_event_includes_latency_when_given():
    event = build_decision_event(2, Placement(rotation=1, col=3, score=0.5), latency_ms=1234.56)
    assert event["data"]["latency_ms"] == 1234.6
    assert "late" not in event["data"]


def test_decision_event_marks_late_only_when_true():
    event = build_decision_event(2, Placement(rotation=1, col=3, score=0.5), latency_ms=20000.0, late=True)
    assert event["data"]["late"] is True


def test_collector_decision_accepts_turn_override_for_late_arrivals():
    seen = []

    class Capture:
        def publish(self, event):
            seen.append(event)

    collector = EventCollector(Capture())
    collector.spawn("J", "O")
    collector.spawn("O", "T")
    collector.decision(Placement(rotation=0, col=4, score=0.0), latency_ms=18000.0, late=True, turn=1)
    assert seen[-1]["turn"] == 1
    assert seen[-1]["data"]["late"] is True
    collector.decision(Placement(rotation=0, col=4, score=0.0))
    assert seen[-1]["turn"] == 2


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


def test_decision_event_includes_tokens_only_when_given():
    plain = build_decision_event(2, Placement(rotation=1, col=3, score=0.5), latency_ms=10.0)
    assert "tokens" not in plain["data"]
    counted = build_decision_event(2, Placement(rotation=1, col=3, score=0.5), latency_ms=10.0, tokens=180)
    assert counted["data"]["tokens"] == 180


def test_graded_event_carries_the_grade_and_leaves_the_decision_event_alone():
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.events import build_decision_event, build_graded_event
    from tetris_agent.policy import Placement

    board = np.zeros((18, 10), dtype=bool)
    board[15:18, :6] = True
    chosen = Placement(rotation=0, col=7, score=0.0)
    g = quality.grade(board, "O", "I", chosen)

    event = build_graded_event(turn=4, grade=g)
    assert event["event_type"] == "placement_graded"
    assert event["turn"] == 4
    assert event["data"]["rank"] == g.rank
    assert event["data"]["regret_norm"] == round(g.regret_norm, 6)

    # The decision event must be byte-identical to before this feature.
    decision = build_decision_event(turn=4, placement=chosen)
    assert set(decision["data"]) == {"rotation", "col", "score"}


def test_collector_publishes_a_graded_event():
    import numpy as np

    from tetris_agent import quality
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Placement

    published = []

    class Sink:
        def publish(self, event):
            published.append(event)

    board = np.zeros((18, 10), dtype=bool)
    board[15:18, :6] = True
    collector = EventCollector(Sink())
    collector.graded(quality.grade(board, "O", "I", Placement(rotation=0, col=7, score=0.0)))
    assert [e["event_type"] for e in published] == ["placement_graded"]
