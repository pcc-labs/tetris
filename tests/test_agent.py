import json
from types import SimpleNamespace

import pytest

from tetris_agent.cli import build_config
from tetris_agent.policy import Placement


def test_build_config_defaults():
    cfg = build_config([])
    assert cfg.rom == "rom/tetris.gb"
    assert cfg.max_pieces == 300
    assert cfg.timer_div is None
    assert cfg.output_json is None
    assert cfg.telemetry_dir == "data/telemetry"
    assert cfg.runs_dir == "runs"
    assert not cfg.no_telemetry and not cfg.no_record and not cfg.no_self_heal


def test_build_config_flags():
    cfg = build_config(
        ["--rom", "x.gb", "--max-pieces", "50", "--timer-div", "0", "--no-self-heal", "--no-record", "--label", "demo"]
    )
    assert cfg.rom == "x.gb"
    assert cfg.max_pieces == 50
    assert cfg.timer_div == 0
    assert cfg.no_self_heal and cfg.no_record
    assert cfg.label == "demo"


@pytest.mark.rom
def test_agent_places_pieces_and_records(rom_path, tmp_path):
    from tetris_agent.agent import TetrisAgent
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Genome
    from tetris_agent.publisher import NoopPublisher
    from tetris_agent.recorder import RunRecorder

    emu = Emulator(rom_path)
    recorder = RunRecorder(tmp_path, label="test")
    agent = TetrisAgent(
        emu,
        genome=Genome(),
        collector=EventCollector(NoopPublisher()),
        recorder=recorder,
        max_pieces=5,
    )
    try:
        fitness = agent.run(timer_div=0x00)
    finally:
        emu.stop()
    assert fitness["pieces_placed"] == 5
    assert fitness["topped_out"] is False
    run_dir = tmp_path / recorder.run_id
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["fitness"] == fitness
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    spawns = [e for e in events if e["event_type"] == "piece_spawn"]
    assert len(spawns) == 5


def test_build_config_live_flag():
    cfg = build_config(["--live"])
    assert cfg.live is True
    assert build_config([]).live is False


def test_build_config_policy_defaults_and_overrides():
    cfg = build_config([])
    assert cfg.policy == "heuristic"
    assert cfg.harness == "features"
    cfg = build_config(["--policy", "model", "--model", "claude-sonnet-5", "--harness", "board", "--effort", "low"])
    assert (cfg.policy, cfg.model, cfg.harness, cfg.effort) == ("model", "claude-sonnet-5", "board", "low")


def test_build_policy_returns_heuristic_without_touching_the_api():
    from tetris_agent.cli import build_policy

    policy = build_policy(build_config([]))
    assert policy.name == "heuristic"
    assert policy.stats()["cost_usd"] == 0.0


def test_build_config_parses_level_for_live_gravity():
    assert build_config(["--live", "--level", "3"]).level == 3
    assert build_config([]).level == 0


def test_agent_class_routes_live_to_the_no_pause_loop():
    from tetris_agent.agent import TetrisAgent
    from tetris_agent.cli import _agent_class
    from tetris_agent.live_agent import LiveTetrisAgent

    assert _agent_class(True) is LiveTetrisAgent
    assert _agent_class(False) is TetrisAgent


# ---- bounded pause: --decision-deadline ------------------------------------


class TickClock:
    """Manual monotonic clock the fake policy advances while 'thinking'."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class SlowPolicy:
    """Thinks for a scripted number of clock-seconds, then answers legally."""

    name = "slow"

    def __init__(self, clock, think_s):
        self.clock = clock
        self.think_s = think_s
        self.last_reason = ""
        self.deadline_s = None

    def plan(self, board, piece, next_piece, turn):
        from tetris_agent.policy import Placement

        self.clock.advance(self.think_s)
        return Placement(rotation=0, col=4, score=0.0)

    def stats(self):
        return {"policy": self.name}


class ListPublisher:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


def _paused_agent_fixtures(
    monkeypatch, think_s, deadline_s, max_pieces=1, session_meta=None, policy=None, grader=None, clock=None
):
    import numpy as np

    from tetris_agent.agent import TetrisAgent
    from tetris_agent.controller import ExecResult
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Genome

    clock = clock or TickClock()
    calls = []

    class FakeController:
        def __init__(self, emu, ticks_per_press=4):
            pass

        def execute(self, target):
            calls.append("execute")
            return ExecResult(locked=True, misexec=0, lines_delta=0, replanned=False)

        def run_out(self):
            calls.append("run_out")
            return ExecResult(locked=True, misexec=0, lines_delta=0, replanned=False)

    class FakeState:
        def __init__(self, falling):
            self.falling = falling
            self.board = np.zeros((18, 10), dtype=bool)
            self.next_piece = "I"
            self.game_over = False

    class FakeFalling:
        name = "O"

    class FakeEmu:
        score = 0
        lines = 0
        level = 0

        def start(self, timer_div=None):
            pass

        def tick(self, n):
            pass

    states = []
    for _ in range(max_pieces):
        states += [FakeState(FakeFalling()), FakeState(None)]

    def fake_read_state(emu):
        return states.pop(0) if len(states) > 1 else states[0]

    monkeypatch.setattr("tetris_agent.agent.Controller", FakeController)
    monkeypatch.setattr("tetris_agent.agent.read_state", fake_read_state)

    publisher = ListPublisher()
    policy = policy or SlowPolicy(clock, think_s=think_s)
    agent = TetrisAgent(
        FakeEmu(),
        genome=Genome(),
        collector=EventCollector(publisher),
        max_pieces=max_pieces,
        policy=policy,
        decision_deadline_s=deadline_s,
        session_meta=session_meta,
        clock=clock,
        grader=grader,
    )
    return agent, calls, publisher


def test_paused_deadline_discards_the_slow_decision_and_lets_the_piece_fall(monkeypatch):
    agent, calls, publisher = _paused_agent_fixtures(monkeypatch, think_s=20, deadline_s=15.0)
    fitness = agent.run(timer_div=0)

    assert calls == ["run_out"]  # never executed the stale plan
    assert agent.paused_stats["timeouts"] == 1
    assert agent.paused_stats["decision_latencies_ms"] == [20000.0]
    decision = next(e for e in publisher.events if e["event_type"] == "placement_decision")
    assert decision["data"]["late"] is True
    assert decision["data"]["latency_ms"] == 20000.0
    assert fitness["policy"]["timeouts"] == 1


def test_paused_deadline_executes_decisions_that_beat_the_clock(monkeypatch):
    agent, calls, publisher = _paused_agent_fixtures(monkeypatch, think_s=5, deadline_s=15.0)
    agent.run(timer_div=0)

    assert calls == ["execute"]
    assert agent.paused_stats["timeouts"] == 0
    assert agent.paused_stats["decision_latencies_ms"] == [5000.0]
    assert agent.policy.deadline_s == 15.0  # the policy was told its budget
    decision = next(e for e in publisher.events if e["event_type"] == "placement_decision")
    assert "late" not in decision["data"]


def test_paused_agent_without_deadline_keeps_legacy_behavior(monkeypatch):
    agent, calls, publisher = _paused_agent_fixtures(monkeypatch, think_s=300, deadline_s=None)
    agent.run(timer_div=0)

    assert calls == ["execute"]
    assert agent.paused_stats["timeouts"] == 0


def test_deadline_survives_policies_that_have_no_deadline_attribute(monkeypatch):
    class NoDeadlinePolicy(SlowPolicy):
        def __init__(self, clock, think_s):
            super().__init__(clock, think_s)
            del self.deadline_s

    clock = TickClock()
    agent, calls, _ = _paused_agent_fixtures(monkeypatch, think_s=1, deadline_s=15.0, policy=NoDeadlinePolicy(clock, 1))
    agent.run(timer_div=0)  # must not raise
    assert calls == ["execute"]


def test_session_meta_lands_on_the_session_start_event(monkeypatch):
    meta = {"arm": "pi/gpt-oss:20b/features/medium+p15", "model": "pi/gpt-oss:20b", "seed": 1}
    agent, _, publisher = _paused_agent_fixtures(monkeypatch, think_s=1, deadline_s=15.0, session_meta=meta)
    agent.run(timer_div=1)

    start = next(e for e in publisher.events if e["event_type"] == "session" and e["data"]["phase"] == "start")
    assert start["data"]["model"] == "pi/gpt-oss:20b"
    assert start["data"]["arm"] == "pi/gpt-oss:20b/features/medium+p15"


def test_spawn_is_published_before_the_policy_starts_thinking(monkeypatch):
    """The viewer shows THINKING between spawn and decision; a spawn published
    after plan() returns would leave the freeze unexplained on screen."""
    seen = {}

    class ProbePolicy(SlowPolicy):
        def __init__(self, clock, publisher):
            super().__init__(clock, think_s=1)
            self.publisher = publisher

        def plan(self, board, piece, next_piece, turn):
            seen["spawns_before_plan"] = sum(1 for e in self.publisher.events if e["event_type"] == "piece_spawn")
            seen["turn_arg"] = turn
            return super().plan(board, piece, next_piece, turn)

    clock = TickClock()
    probe = ProbePolicy(clock, None)
    agent, _, publisher = _paused_agent_fixtures(monkeypatch, think_s=1, deadline_s=None, policy=probe)
    probe.publisher = publisher
    agent.run(timer_div=0)

    assert seen["spawns_before_plan"] == 1
    assert seen["turn_arg"] == 1  # turn already advanced by the spawn


def test_the_frozen_board_is_captured_at_spawn_so_watchers_see_it(monkeypatch):
    """During the think-freeze the viewer must show the board being decided on,
    not the tail of the previous piece's animation."""
    frames = []

    class ProbeSinkPolicy(SlowPolicy):
        def plan(self, board, piece, next_piece, turn):
            frames_at_plan.append(len(frames))
            return super().plan(board, piece, next_piece, turn)

    frames_at_plan = []
    clock = TickClock()
    agent, _, _ = _paused_agent_fixtures(monkeypatch, think_s=1, deadline_s=15.0, policy=ProbeSinkPolicy(clock, 1))
    agent.frame_sinks.append(lambda turn, png: frames.append(turn))
    agent.emu.screenshot = lambda: b"png"
    agent.run(timer_div=0)

    assert frames_at_plan == [1]  # exactly one frame captured before thinking began


# ---- placement grading -------------------------------------------------


def _fake_grader(calls, clock=None, grade_s=7, regret=0.25, rank=2):
    """A grader that records what it was asked and, when given a clock,
    advances it — so a test using this grader can tell whether grading's own
    cost was charged to the recorded decision latency."""

    def grade(board, piece, next_piece, placement):
        calls.append((board, piece, placement.rotation, placement.col))
        if clock is not None:
            clock.advance(grade_s)
        return SimpleNamespace(
            regret_norm=regret,
            rank=rank,
            to_dict=lambda: {"rank": rank, "regret_norm": regret},
        )

    return grade


class FallbackPolicy:
    """A policy whose every answer is the deterministic fallback, not a choice."""

    name = "fallback"
    last_reason = ""
    last_fallback = True

    def plan(self, board, piece, next_piece, turn):
        return Placement(rotation=0, col=0, score=0.0)

    def stats(self):
        return {"policy": self.name, "decisions": 1, "cost_usd": 0.0}


def test_grading_is_not_charged_to_the_model(monkeypatch):
    """Recorded latency must be the think time and nothing else.

    The fake grader advances the shared clock by 7s, the same way SlowPolicy
    advances it while "thinking". If grading ran before the latency
    measurement (or the measurement spanned both), the recorded latency
    would include those 7s; without a clock-advancing grader this test could
    not fail for the reason it names, since 3000.0 would come back either way.
    """
    calls = []
    clock = TickClock()
    agent, _, publisher = _paused_agent_fixtures(
        monkeypatch, think_s=3, deadline_s=15.0, grader=_fake_grader(calls, clock=clock), clock=clock
    )
    agent.run(timer_div=0)
    decision = next(e for e in publisher.events if e["event_type"] == "placement_decision")
    assert decision["data"]["latency_ms"] == 3000.0
    assert calls, "the grader never ran, so this proves nothing"


def test_the_graded_event_follows_the_lock(monkeypatch):
    agent, _, publisher = _paused_agent_fixtures(monkeypatch, think_s=3, deadline_s=15.0, grader=_fake_grader([]))
    agent.run(timer_div=0)
    kinds = [e["event_type"] for e in publisher.events]
    assert kinds.index("piece_locked") < kinds.index("placement_graded")


def test_a_late_decision_is_not_graded(monkeypatch):
    """Gravity placed the piece; the model's choice never ran."""
    calls = []
    agent, _, publisher = _paused_agent_fixtures(monkeypatch, think_s=20, deadline_s=15.0, grader=_fake_grader(calls))
    fitness = agent.run(timer_div=0)
    assert calls == []
    assert "placement_graded" not in [e["event_type"] for e in publisher.events]
    assert fitness["graded_decisions"] == 0
    assert fitness["mean_regret"] is None


def test_a_fallback_placement_is_not_graded(monkeypatch):
    calls = []
    agent, _, _ = _paused_agent_fixtures(
        monkeypatch, think_s=3, deadline_s=15.0, policy=FallbackPolicy(), grader=_fake_grader(calls)
    )
    fitness = agent.run(timer_div=0)
    assert calls == []
    assert fitness["graded_decisions"] == 0


def test_the_graded_event_carries_the_pre_decision_board(monkeypatch):
    calls = []
    agent, _, publisher = _paused_agent_fixtures(monkeypatch, think_s=3, deadline_s=15.0, grader=_fake_grader(calls))
    agent.run(timer_div=0)
    graded = next(e for e in publisher.events if e["event_type"] == "placement_graded")
    rows = graded["data"]["board"]
    assert len(rows) == 18 and all(len(r) == 10 and set(r) <= {".", "#"} for r in rows)
    expected = ["".join("#" if c else "." for c in row) for row in calls[0][0]]
    assert rows == expected
