"""Human baseline: you play the ROM, scored exactly like any other arm.

The benchmark's other reference arms answer narrow questions — `no-input` is
the floor, `random` is chance. Neither tells you what a person scores on the
same fifty pieces, which is the comparison that makes "can a model do this"
mean anything.

So this opens a real window, runs at real speed, and takes no input of its
own: you play. It watches for the same piece-lock events the agent loop does
and feeds them to the same FitnessTracker, so the row it writes is directly
comparable to a model arm's — same seed, same piece sequence, same metrics.
"""

import logging

from tetris_agent.board import features
from tetris_agent.fitness import FitnessTracker
from tetris_agent.state import read_state

logger = logging.getLogger(__name__)

# Enough frames for a piece to fall from spawn to lock even if the player
# never touches a key: ~1 frame per tick at real speed.
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


def play(rom_path, max_pieces: int = 50, timer_div: int = 0) -> dict:
    """Play by hand; returns the same fitness dict a policy arm produces."""
    from tetris_agent.emulator import Emulator

    emu = Emulator(rom_path, headless=False, speed=1)
    tracker = FitnessTracker()
    placed = 0
    closed = False

    print(f"\nManual run — seed {timer_div}, up to {max_pieces} pieces.\n\n{CONTROLS}\n")
    try:
        emu.start(timer_div=timer_div)
        while placed < max_pieces:
            # A piece is in play...
            state = _wait_until(emu, lambda s: s.falling is not None or s.game_over)
            if state is None:
                closed = True
                break
            if state.game_over:
                tracker.on_game_over()
                break
            # ...and now it has locked (or the game ended under it).
            state = _wait_until(emu, lambda s: s.falling is None or s.game_over)
            if state is None:
                closed = True
                break
            # misexec is 0 by definition: a human's presses are the intent.
            tracker.on_lock(features(state.board), misexec=0)
            placed += 1
            print(f"\r  piece {placed}/{max_pieces}   score {emu.score}   lines {emu.lines}", end="", flush=True)
            if state.game_over:
                tracker.on_game_over()
                break
    except KeyboardInterrupt:
        print("\n  stopped early.")
    finally:
        fitness = tracker.compute(score=emu.score, lines=emu.lines, level=emu.level)
        emu.stop()

    if closed:
        print("\n  window closed.")
    return fitness
