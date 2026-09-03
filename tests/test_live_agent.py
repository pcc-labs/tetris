"""LiveTetrisAgent: gravity keeps running while the policy thinks.

The worker thread and the controller are both replaced with hand-driven
fakes, so every timing scenario (fast, slow, never) is scripted — no real
threads, no sleeps.
"""

from types import SimpleNamespace

from test_manual import CapturingPublisher, FakeEmulator, falling_at, state_with

from tetris_agent.controller import ExecResult
from tetris_agent.events import EventCollector
from tetris_agent.live_agent import LiveTetrisAgent
from tetris_agent.policy import Genome, Placement


class LiveFakeEmulator(FakeEmulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_level = None
        self.after_tick = None

    def force_level(self, level):
        self.forced_level = level

    def tick(self, n=1):
        alive = super().tick(n)
        if self.after_tick is not None:
            self.after_tick()
        return alive


class ManualThreads:
    """thread_factory seam: worker targets run when the test says so."""

    def __init__(self, immediate=True):
        self.immediate = immediate
        self.pending = []

    def __call__(self, target=None, daemon=None, name=None):
        runner = self

        class _Thread:
            def start(self):
                if runner.immediate:
                    target()
                else:
                    runner.pending.append(target)

        return _Thread()

    def run_next(self):
        self.pending.pop(0)()


class ScriptedPolicy:
    def __init__(self, placements):
        self.name = "scripted"
        self.placements = list(placements)
        self.calls = []
        self.deadlines = []
        self.last_reason = "scripted"

    def plan(self, board, piece, next_piece, turn):
        self.calls.append((piece, turn))
        self.deadlines.append(getattr(self, "deadline_s", None))
        return self.placements.pop(0)

    def stats(self):
        return {"policy": self.name, "decisions": len(self.calls), "cost_usd": 0.0}


def install_live(monkeypatch, emu):
    monkeypatch.setattr("tetris_agent.live_agent.read_state", lambda _e: emu.current())


def install_controller(monkeypatch):
    """Replace Controller with a stub that records calls and rides to the lock."""
    executed = []

    class _Controller:
        def __init__(self, emu, ticks_per_press=4):
            self.emu = emu

        def execute(self, placement):
            executed.append((placement, self.emu.ticks))
            for _ in range(200):
                if self.emu.current().falling is None or not self.emu.tick(2):
                    break
            return ExecResult(locked=True, misexec=0, lines_delta=0, replanned=False)

    monkeypatch.setattr("tetris_agent.live_agent.Controller", _Controller)
    return executed


def make_agent(emu, policy, pub, threads, max_pieces=2, grader=None):
    return LiveTetrisAgent(
        emu,
        Genome(),
        EventCollector(pub),
        max_pieces=max_pieces,
        policy=policy,
        thread_factory=threads,
        grader=grader,
    )


def test_fast_decision_emits_spawn_decision_locked_per_turn(monkeypatch):
    timeline = [
        state_with(falling_at(0, 3, name="J")),
        state_with(None, filled=4),
        state_with(falling_at(0, 3, name="T")),
        state_with(None, filled=8),
    ]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    executed = install_controller(monkeypatch)
    pub = CapturingPublisher()
    policy = ScriptedPolicy([Placement(0, 0, 0.0), Placement(1, 7, 0.0)])

    fitness = make_agent(emu, policy, pub, ManualThreads(immediate=True)).run(timer_div=0)

    assert [e["turn"] for e in pub.of_type("piece_spawn")] == [1, 2]
    decisions = pub.of_type("placement_decision")
    assert [e["turn"] for e in decisions] == [1, 2]
    assert [e["turn"] for e in pub.of_type("piece_locked")] == [1, 2]
    assert all("latency_ms" in d["data"] for d in decisions)
    assert all("late" not in d["data"] for d in decisions)
    assert [(p.rotation, p.col) for p, _ in executed] == [(0, 0), (1, 7)]
    assert fitness["pieces_placed"] == 2
    assert fitness["policy"]["late"] == 0
    assert fitness["policy"]["in_flight_at_end"] is False


def test_gravity_keeps_running_while_the_model_thinks(monkeypatch):
    # The piece stays falling for 30 timeline steps; the decision resolves
    # only after the 10th tick. The frozen loop would show 0 ticks here.
    timeline = [state_with(falling_at(0, 3))] * 30 + [state_with(None, filled=4)]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    executed = install_controller(monkeypatch)
    threads = ManualThreads(immediate=False)
    tick_calls = {"n": 0}

    def after_tick():
        tick_calls["n"] += 1
        if tick_calls["n"] == 10 and threads.pending:
            threads.run_next()

    emu.after_tick = after_tick
    policy = ScriptedPolicy([Placement(0, 5, 0.0)])

    fitness = make_agent(emu, policy, CapturingPublisher(), threads, max_pieces=1).run()

    assert executed, "the decision should still execute once it arrives"
    _, ticks_at_execute = executed[0]
    assert ticks_at_execute >= 20
    assert fitness["pieces_placed"] == 1


def test_window_closed_mid_thought_returns_clean_fitness(monkeypatch):
    timeline = [state_with(falling_at(0, 3))] * 50
    emu = LiveFakeEmulator(timeline, alive_for=6)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    pub = CapturingPublisher()

    fitness = make_agent(emu, ScriptedPolicy([]), pub, ManualThreads(immediate=False), max_pieces=5).run()

    assert fitness["pieces_placed"] == 0
    assert fitness["policy"]["in_flight_at_end"] is True
    assert pub.of_type("game_over")


def test_late_decision_is_discarded_and_the_next_piece_gets_a_fresh_one(monkeypatch):
    # Piece J locks while its decision is still in flight; T spawns; the J
    # decision then arrives (stale), is discarded as late, and a fresh
    # request goes out for T, which executes.
    timeline = (
        [state_with(falling_at(0, 3, name="J"))] * 2
        + [state_with(None, filled=4)]
        + [state_with(falling_at(0, 3, name="T"))] * 8
        + [state_with(None, filled=8)]
    )
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    executed = install_controller(monkeypatch)
    threads = ManualThreads(immediate=False)
    tick_calls = {"n": 0}

    def after_tick():
        tick_calls["n"] += 1
        if tick_calls["n"] in (6, 8) and threads.pending:
            threads.run_next()

    emu.after_tick = after_tick
    pub = CapturingPublisher()
    policy = ScriptedPolicy([Placement(0, 0, 0.0), Placement(1, 7, 0.0)])

    fitness = make_agent(emu, policy, pub, threads).run()

    late = [e for e in pub.of_type("placement_decision") if e["data"].get("late")]
    fresh = [e for e in pub.of_type("placement_decision") if not e["data"].get("late")]
    assert [e["turn"] for e in late] == [1]
    assert [e["turn"] for e in fresh] == [2]
    assert [(p.rotation, p.col) for p, _ in executed] == [(1, 7)]
    assert [piece for piece, _ in policy.calls] == ["J", "T"]
    assert fitness["policy"]["late"] == 1
    assert fitness["pieces_placed"] == 2


def test_piece_locking_without_a_decision_still_counts(monkeypatch):
    timeline = [state_with(falling_at(0, 3, name="J"))] * 2 + [state_with(None, filled=4)]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    executed = install_controller(monkeypatch)
    pub = CapturingPublisher()

    fitness = make_agent(emu, ScriptedPolicy([]), pub, ManualThreads(immediate=False), max_pieces=1).run()

    locked = pub.of_type("piece_locked")
    assert [e["turn"] for e in locked] == [1]
    assert locked[0]["data"]["misexec"] == 0
    assert executed == []
    assert fitness["pieces_placed"] == 1
    assert fitness["policy"]["in_flight_at_end"] is True


def test_decision_arriving_just_after_its_own_piece_locked_is_late(monkeypatch):
    # The decision resolves during the very tick the piece locks on: same
    # turn, but falling is already None — it must not reach the controller.
    timeline = [state_with(falling_at(0, 3, name="J"))] * 2 + [state_with(None, filled=4)]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    executed = install_controller(monkeypatch)
    threads = ManualThreads(immediate=False)
    tick_calls = {"n": 0}

    def after_tick():
        tick_calls["n"] += 1
        if tick_calls["n"] == 2 and threads.pending:
            threads.run_next()

    emu.after_tick = after_tick
    pub = CapturingPublisher()
    policy = ScriptedPolicy([Placement(0, 0, 0.0)])

    fitness = make_agent(emu, policy, pub, threads, max_pieces=1).run()

    assert executed == []
    assert fitness["policy"]["late"] == 1
    late = [e for e in pub.of_type("placement_decision") if e["data"].get("late")]
    assert [e["turn"] for e in late] == [1]
    assert [e["turn"] for e in pub.of_type("piece_locked")] == [1]
    assert fitness["pieces_placed"] == 1


class RaisingPolicy(ScriptedPolicy):
    def plan(self, board, piece, next_piece, turn):
        self.calls.append((piece, turn))
        raise RuntimeError("api fell over")


def test_plan_returning_none_ends_the_run_as_top_out(monkeypatch):
    emu = LiveFakeEmulator([state_with(falling_at(0, 3, name="J"))] * 3)
    install_live(monkeypatch, emu)
    executed = install_controller(monkeypatch)
    pub = CapturingPublisher()

    fitness = make_agent(emu, ScriptedPolicy([None]), pub, ManualThreads(immediate=True), max_pieces=5).run()

    assert executed == []
    assert fitness["topped_out"] is True
    stuck = pub.of_type("stuck")
    assert any("no placement fits" in e["data"]["detail"] for e in stuck)


def test_worker_exception_falls_back_to_first_legal_move(monkeypatch):
    timeline = [state_with(falling_at(0, 3, name="J"))] * 2 + [state_with(None, filled=4)]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    executed = install_controller(monkeypatch)
    pub = CapturingPublisher()

    fitness = make_agent(emu, RaisingPolicy([]), pub, ManualThreads(immediate=True), max_pieces=1).run()

    assert len(executed) == 1
    decision = pub.of_type("placement_decision")[0]
    assert decision["data"]["reason"] == "worker error fallback"
    assert "latency_ms" in decision["data"]
    assert fitness["policy"]["worker_errors"] == 1
    assert fitness["pieces_placed"] == 1


def test_game_over_with_a_decision_still_in_flight_ends_cleanly(monkeypatch):
    timeline = [state_with(falling_at(0, 3, name="J"))] * 2 + [state_with(None, filled=4, game_over=True)]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    pub = CapturingPublisher()

    fitness = make_agent(emu, ScriptedPolicy([]), pub, ManualThreads(immediate=False), max_pieces=5).run()

    assert fitness["topped_out"] is True
    assert fitness["policy"]["in_flight_at_end"] is True
    assert fitness["pieces_placed"] == 0
    assert pub.of_type("game_over")


def test_force_level_pins_gravity_by_default(monkeypatch):
    emu = LiveFakeEmulator([state_with(falling_at(0, 3)), state_with(None, filled=4)])
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)

    make_agent(emu, ScriptedPolicy([Placement(0, 3, 0.0)]), CapturingPublisher(), ManualThreads(), max_pieces=1).run()

    assert emu.forced_level == 0


def test_force_level_none_leaves_gravity_alone(monkeypatch):
    emu = LiveFakeEmulator([state_with(falling_at(0, 3)), state_with(None, filled=4)])
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)

    make_agent(emu, ScriptedPolicy([Placement(0, 3, 0.0)]), CapturingPublisher(), ManualThreads(), max_pieces=1).run(
        level=None
    )

    assert emu.forced_level is None


def test_force_level_override_is_honored(monkeypatch):
    emu = LiveFakeEmulator([state_with(falling_at(0, 3)), state_with(None, filled=4)])
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)

    make_agent(emu, ScriptedPolicy([Placement(0, 3, 0.0)]), CapturingPublisher(), ManualThreads(), max_pieces=1).run(
        level=5
    )

    assert emu.forced_level == 5


import pytest


@pytest.mark.rom
def test_live_agent_places_pieces_on_the_real_rom(rom_path):
    # Real threads, real emulator. speed=0 keeps the pacer off so the test is
    # fast; the loop semantics are identical to a paced live run.
    from tetris_agent.emulator import Emulator
    from tetris_agent.policy import HeuristicPolicy

    pub = CapturingPublisher()
    emu = Emulator(rom_path)
    try:
        agent = LiveTetrisAgent(emu, Genome(), EventCollector(pub), max_pieces=3, policy=HeuristicPolicy(Genome()))
        fitness = agent.run(timer_div=0)
    finally:
        emu.stop()

    assert fitness["pieces_placed"] == 3
    spawns = {e["turn"]: i for i, e in enumerate(pub.events) if e["event_type"] == "piece_spawn"}
    decisions = {e["turn"]: i for i, e in enumerate(pub.events) if e["event_type"] == "placement_decision"}
    for turn, spawn_idx in spawns.items():
        if turn in decisions:
            assert spawn_idx < decisions[turn]


def test_deadline_is_set_from_the_piece_height_before_each_submit(monkeypatch):
    # Spawn-height piece (bottom row 2): 15 rows to fall at level 0 =
    # 15 * 53/60 s, minus 2s execution headroom -> 11.25s. The resubmitted
    # piece sits at row 11: 6 rows -> 6 * 53/60 - 2 = 3.3s.
    from tetris_agent.state import FallingPiece

    low_piece = FallingPiece(name="T", rotation=0, cells=frozenset({(10, 4), (10, 5), (10, 6), (11, 5)}))
    timeline = (
        [state_with(falling_at(0, 3, name="J"))] * 2
        + [state_with(None, filled=4)]
        + [state_with(low_piece)] * 8
        + [state_with(None, filled=8)]
    )
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    threads = ManualThreads(immediate=False)
    tick_calls = {"n": 0}

    def after_tick():
        tick_calls["n"] += 1
        if tick_calls["n"] in (6, 8) and threads.pending:
            threads.run_next()

    emu.after_tick = after_tick
    policy = ScriptedPolicy([Placement(0, 0, 0.0), Placement(0, 4, 0.0)])

    make_agent(emu, policy, CapturingPublisher(), threads).run()

    assert policy.deadlines == pytest.approx([11.25, 3.3])


def test_decision_event_carries_the_calls_output_tokens(monkeypatch):
    timeline = [state_with(falling_at(0, 3, name="J")), state_with(None, filled=4)]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    pub = CapturingPublisher()
    policy = ScriptedPolicy([Placement(0, 3, 0.0)])
    policy.last_output_tokens = 42

    make_agent(emu, policy, pub, ManualThreads(immediate=True), max_pieces=1).run()

    decision = pub.of_type("placement_decision")[0]
    assert decision["data"]["tokens"] == 42


def _fake_grader(calls):
    def grade(board, piece, next_piece, placement):
        calls.append((piece, next_piece, placement.rotation, placement.col))
        return SimpleNamespace(regret_norm=0.25, rank=2, to_dict=lambda: {"rank": 2, "regret_norm": 0.25})

    return grade


def test_live_grades_an_executed_decision_after_the_lock(monkeypatch):
    """Order must read spawn -> decision -> locked -> graded."""
    timeline = [
        state_with(falling_at(0, 3, name="J")),
        state_with(None, filled=4),
    ]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    pub = CapturingPublisher()
    calls = []
    agent = make_agent(
        emu,
        ScriptedPolicy([Placement(0, 0, 0.0)]),
        pub,
        ManualThreads(immediate=True),
        max_pieces=1,
        grader=_fake_grader(calls),
    )
    agent.run(timer_div=0)

    kinds = [e["event_type"] for e in pub.events]
    assert "placement_graded" in kinds
    assert kinds.index("piece_locked") < kinds.index("placement_graded")
    # The board and next piece handed to the grader are the ones the decision saw.
    assert calls == [("J", "I", 0, 0)]


def test_a_piece_that_locks_without_a_decision_is_not_graded(monkeypatch):
    """The worker never answered, so gravity placed the piece and there is
    nothing to grade."""
    timeline = [
        state_with(falling_at(0, 3, name="J")),
        state_with(None, filled=4),
    ]
    emu = LiveFakeEmulator(timeline)
    install_live(monkeypatch, emu)
    install_controller(monkeypatch)
    pub = CapturingPublisher()
    calls = []
    agent = make_agent(
        emu,
        ScriptedPolicy([Placement(0, 0, 0.0)]),
        pub,
        ManualThreads(immediate=False),  # the decision is never delivered
        max_pieces=1,
        grader=_fake_grader(calls),
    )
    agent.run(timer_div=0)

    assert calls == []
    assert "placement_graded" not in [e["event_type"] for e in pub.events]
