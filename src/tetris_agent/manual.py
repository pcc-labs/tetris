"""Human baseline: you play the ROM, recorded exactly like any other arm.

The benchmark's other reference arms answer narrow questions — `no-input` is
the floor, `random` is chance. Neither tells you what a person scores on the
same fifty pieces, which is the comparison that makes "can a model do this"
mean anything.

So this opens a real window, runs at real speed, and takes no input of its
own: you play. It watches the same falling→locked transitions the agent loop
does and emits the same events, so a manual run produces the same artefacts a
model run does — `runs/<id>/events.jsonl` with one spawn/decision/locked
triple per piece, and a `summary.json` with the outcome. The viewer replays it
without knowing a human was driving.

The one difference is where the placement comes from. A model arm emits the
placement it *chose*; here we emit the placement we *observed*, read off the
falling piece on the last frame before it locked.
"""

import logging

from tetris_agent.board import features
from tetris_agent.fitness import FitnessTracker
from tetris_agent.policy import Placement
from tetris_agent.state import read_state

logger = logging.getLogger(__name__)

# Enough frames for a piece to fall from spawn to lock even if the player
# never touches a key.
_WAIT_LIMIT = 60 * 60 * 5

CONTROLS = """\
  arrow keys   move / soft drop        a   rotate
  return       start                   s   rotate (other button)
  close window or press ctrl-c to stop early"""


def _wait_until(emu, predicate, limit: int = _WAIT_LIMIT):
    """Tick until predicate(state) holds; returns that state, or None if the
    window closed or we ran out of patience."""
    for _ in range(limit):
        state = read_state(emu)
        if predicate(state):
            return state
        if not emu.tick(2):
            return None
    return None


def _follow_until_locked(emu, limit: int = _WAIT_LIMIT):
    """Track a falling piece to its resting place.

    Returns (last_state_with_the_piece, ended) where `ended` is "locked",
    "game_over", or "closed". The last state carries the rotation and column
    the piece came to rest at — the human's placement, in the same terms a
    model arm reports.
    """
    last_seen = read_state(emu)
    for _ in range(limit):
        state = read_state(emu)
        if state.game_over:
            return last_seen, "game_over"
        if state.falling is None:
            return last_seen, "locked"
        last_seen = state
        if not emu.tick(2):
            return last_seen, "closed"
    return last_seen, "locked"


def observe_loop(emu, collector, tracker, max_pieces: int) -> tuple[int, bool]:
    """Watch falling→locked transitions and emit spawn/decision/locked events.

    Shared by every human-driven surface (SDL window, browser): the input
    source differs, the observation doesn't. Returns (pieces placed, whether
    the window/connection closed early).
    """
    placed = 0
    closed = False
    lines_before = 0
    try:
        while placed < max_pieces:
            state = _wait_until(emu, lambda s: s.falling is not None or s.game_over)
            if state is None:
                closed = True
                break
            if state.game_over:
                tracker.on_game_over()
                break

            collector.spawn(state.falling.name, state.next_piece)
            last_seen, ended = _follow_until_locked(emu)
            if ended == "closed":
                closed = True
                break

            # Where the human actually put it — same (rotation, col) vocabulary
            # a model arm uses, so the two are directly comparable.
            if last_seen.falling is not None:
                collector.decision(
                    Placement(rotation=last_seen.falling.rotation, col=last_seen.falling.col, score=0.0),
                    reason="human",
                )

            settled = read_state(emu)
            lines_delta = emu.lines - lines_before
            lines_before = emu.lines
            f = features(settled.board)
            # misexec is 0 by definition: a human's presses are the intent.
            tracker.on_lock(f, misexec=0)
            collector.locked(lines_delta, f, misexec=0, score=emu.score)
            placed += 1
            print(
                f"\r  piece {placed}/{max_pieces}   score {emu.score}   lines {emu.lines}   holes {f.holes}   ",
                end="",
                flush=True,
            )
            if ended == "game_over" or settled.game_over:
                tracker.on_game_over()
                break
    except KeyboardInterrupt:
        print("\n  stopped early.")
    return placed, closed


def play(rom_path, max_pieces: int = 50, timer_div: int = 0, collector=None, recorder=None) -> dict:
    """Play by hand; returns the same fitness dict a policy arm produces."""
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.publisher import NoopPublisher

    collector = collector or EventCollector(NoopPublisher())
    emu = Emulator(rom_path, headless=False, speed=1)
    if recorder is not None:
        emu.frame_hook = lambda: recorder.record_frame(collector.turn, emu.screenshot())

    tracker = FitnessTracker()
    placed = 0
    closed = False

    print(f"\nManual run — seed {timer_div}, up to {max_pieces} pieces.\n\n{CONTROLS}\n")
    collector.session("start", {"policy": "human", "timer_div": timer_div, "max_pieces": max_pieces})
    try:
        emu.start(timer_div=timer_div)
        placed, closed = observe_loop(emu, collector, tracker, max_pieces)
    finally:
        fitness = tracker.compute(score=emu.score, lines=emu.lines, level=emu.level)
        fitness["policy"] = {"policy": "human", "decisions": placed, "cost_usd": 0.0}
        collector.game_over(fitness)
        collector.session("end", {"fitness": fitness})
        emu.stop()

    if closed:
        print("\n  window closed.")
    if recorder is not None:
        out = recorder.finalize(fitness=fitness, params={})
        print(f"\n  run recorded to {out}")
    return fitness
