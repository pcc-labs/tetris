import numpy as np

from tetris_agent.manual import play
from tetris_agent.state import FallingPiece, GameState


def board_with(n_filled: int) -> np.ndarray:
    board = np.zeros((18, 10), dtype=bool)
    flat = board.reshape(-1)
    flat[:n_filled] = True
    return board


def state(falling: bool, filled: int = 0, game_over: bool = False) -> GameState:
    piece = FallingPiece(name="O", rotation=0, cells=frozenset({(1, 4), (1, 5), (2, 4), (2, 5)}))
    return GameState(
        board=board_with(filled),
        falling=piece if falling else None,
        next_piece="I",
        score=0,
        level=0,
        lines=0,
        game_over=game_over,
    )


class FakeEmulator:
    """Walks a scripted timeline, advancing on tick — never on read.

    Matters because read_state is a pure read of the current emulator state:
    a fake that mutates per read would let a read-count change alter results.
    """

    def __init__(self, timeline, alive_for=10_000):
        self.timeline = list(timeline)
        self.pos = 0
        self.alive_for = alive_for
        self.ticks = 0
        self.started_with = None
        self.stopped = False
        self.score, self.lines, self.level = 120, 3, 0

    def start(self, timer_div=None):
        self.started_with = timer_div

    def tick(self, n=1):
        self.ticks += n
        self.pos = min(self.pos + 1, len(self.timeline) - 1)
        return self.ticks < self.alive_for

    def stop(self):
        self.stopped = True

    def current(self):
        return self.timeline[self.pos]


def install(monkeypatch, emu):
    monkeypatch.setattr("tetris_agent.emulator.Emulator", lambda *a, **k: emu)
    monkeypatch.setattr("tetris_agent.manual.read_state", lambda _e: emu.current())


def test_counts_one_piece_per_lock_and_reports_fitness(monkeypatch):
    # Each piece is one falling→locked transition on the timeline.
    timeline = []
    for i in range(4):
        timeline += [state(True, filled=i * 4), state(False, filled=(i + 1) * 4)]
    emu = FakeEmulator(timeline)
    install(monkeypatch, emu)

    fitness = play("rom.gb", max_pieces=4, timer_div=7)

    assert fitness["pieces_placed"] == 4
    assert fitness["score"] == 120
    assert fitness["lines"] == 3
    assert emu.started_with == 7  # same seed plumbing as the model arms
    assert emu.stopped


def test_stops_and_flags_game_over(monkeypatch):
    emu = FakeEmulator([state(True), state(False), state(False, game_over=True)])
    install(monkeypatch, emu)

    fitness = play("rom.gb", max_pieces=50)

    assert fitness["topped_out"] is True
    assert fitness["pieces_placed"] < 50


def test_closing_the_window_ends_the_run_cleanly(monkeypatch):
    # tick() goes False when the player closes the window; we should still
    # return a usable fitness row rather than hanging or raising.
    emu = FakeEmulator([state(False)] * 50, alive_for=6)
    install(monkeypatch, emu)

    fitness = play("rom.gb", max_pieces=50)

    assert fitness["pieces_placed"] == 0
    assert emu.stopped


class CapturingPublisher:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)

    def of_type(self, kind):
        return [e for e in self.events if e.get("event_type") == kind]


def falling_at(rotation, col, name="T"):
    cells = frozenset({(1, col), (1, col + 1), (1, col + 2), (2, col + 1)})
    return FallingPiece(name=name, rotation=rotation, cells=cells)


def state_with(piece, filled=0, game_over=False):
    return GameState(
        board=board_with(filled),
        falling=piece,
        next_piece="I",
        score=0,
        level=0,
        lines=0,
        game_over=game_over,
    )


def test_records_which_block_where_it_ended_and_the_outcome(monkeypatch):
    from tetris_agent.events import EventCollector

    # One piece: an L rotated to 2, resting with its left edge on column 6.
    timeline = [state_with(falling_at(2, 6, name="L")), state_with(None, filled=4)]
    emu = FakeEmulator(timeline)
    install(monkeypatch, emu)
    pub = CapturingPublisher()

    play("rom.gb", max_pieces=1, timer_div=0, collector=EventCollector(pub))

    spawn = pub.of_type("piece_spawn")[0]
    assert spawn["data"]["piece"] == "L"  # which block
    assert spawn["data"]["next_piece"] == "I"

    decision = pub.of_type("placement_decision")[0]
    assert (decision["data"]["rotation"], decision["data"]["col"]) == (2, 6)  # where it ended
    assert decision["data"]["reason"] == "human"

    locked = pub.of_type("piece_locked")[0]
    assert "holes" in locked["data"]  # the outcome of putting it there
    assert pub.of_type("game_over")  # and the outcome of the run


def test_emits_one_triple_per_piece(monkeypatch):
    from tetris_agent.events import EventCollector

    timeline = []
    for i in range(3):
        timeline += [state_with(falling_at(0, i)), state_with(None, filled=(i + 1) * 4)]
    emu = FakeEmulator(timeline)
    install(monkeypatch, emu)
    pub = CapturingPublisher()

    play("rom.gb", max_pieces=3, timer_div=0, collector=EventCollector(pub))

    assert len(pub.of_type("piece_spawn")) == 3
    assert len(pub.of_type("placement_decision")) == 3
    assert len(pub.of_type("piece_locked")) == 3
    assert [d["data"]["col"] for d in pub.of_type("placement_decision")] == [0, 1, 2]
