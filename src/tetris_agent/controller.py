"""Execute a planned placement with per-press verification.

This is the in-run healing layer: every rotate/shift press is checked against
the emulator's actual state, unverified presses are counted as misexecutions,
and a piece that drifts (wall, wedge, early lock) surfaces as replanned=True
so the agent can emit a stuck signal instead of silently diverging.
"""

from dataclasses import dataclass

from tetris_agent.policy import Placement
from tetris_agent.state import GameState, read_state

_MAX_ROTATE_PRESSES = 6
_MAX_SHIFT_UNVERIFIED = 3
_MAX_DROP_PRESSES = 220
_MAX_SPAWN_WAIT = 150
# Level-0 gravity is ~52 frames/row over 18 rows (~940 frames spawn-to-floor);
# 2000 ticks of 4 frames covers that several times over at any level.
_MAX_RUNOUT_TICKS = 2000


@dataclass
class ExecResult:
    locked: bool
    misexec: int
    lines_delta: int
    replanned: bool


class Controller:
    def __init__(self, emu, ticks_per_press: int = 4):
        self.emu = emu
        self.ticks_per_press = ticks_per_press

    def execute(self, target: Placement) -> ExecResult:
        misexec = 0
        replanned = False
        lines_before = self.emu.lines
        state = read_state(self.emu)
        if state.falling is None:
            return ExecResult(locked=False, misexec=0, lines_delta=0, replanned=True)
        piece_type = state.falling.name

        # Rotate: the A button decrements the rotation value (verified against ROM).
        for _ in range(_MAX_ROTATE_PRESSES):
            state = read_state(self.emu)
            if self._lost_piece(state, piece_type):
                return self._locked_result(misexec, lines_before, replanned=True)
            if state.falling.rotation == target.rotation:
                break
            before = state.falling.rotation
            self.emu.press("a", ticks_after=self.ticks_per_press)
            state = read_state(self.emu)
            if state.falling is not None and state.falling.rotation == before:
                misexec += 1

        # Shift: one verified column per press.
        unverified = 0
        while True:
            state = read_state(self.emu)
            if self._lost_piece(state, piece_type):
                return self._locked_result(misexec, lines_before, replanned=True)
            col = state.falling.col
            if col == target.col:
                break
            button = "left" if target.col < col else "right"
            self.emu.press(button, ticks_after=self.ticks_per_press)
            state = read_state(self.emu)
            if state.falling is None or state.falling.col == col:
                misexec += 1
                unverified += 1
                if unverified >= _MAX_SHIFT_UNVERIFIED:
                    replanned = True
                    break
            else:
                unverified = 0

        # Drop: hold down until the piece joins the settled board.
        settled_before = int(read_state(self.emu).board.sum())
        locked = False
        for _ in range(_MAX_DROP_PRESSES):
            self.emu.press("down", ticks_after=2)
            state = read_state(self.emu)
            if state.game_over:
                locked = True
                break
            if int(state.board.sum()) != settled_before or state.falling is None:
                locked = True
                break
        lines_delta = self._await_spawn(lines_before)
        return ExecResult(locked=locked, misexec=misexec, lines_delta=lines_delta, replanned=replanned)

    def run_out(self) -> ExecResult:
        """No presses: tick until gravity locks the falling piece.

        The bounded-pause path for a decision that blew its deadline — the
        stale plan is discarded and the piece falls where it spawned, so a
        timeout degrades toward no-input instead of executing a late answer.
        """
        lines_before = self.emu.lines
        settled_before = int(read_state(self.emu).board.sum())
        locked = False
        for _ in range(_MAX_RUNOUT_TICKS):
            self.emu.tick(4)
            state = read_state(self.emu)
            if state.game_over:
                locked = True
                break
            if int(state.board.sum()) != settled_before or state.falling is None:
                locked = True
                break
        lines_delta = self._await_spawn(lines_before)
        return ExecResult(locked=locked, misexec=0, lines_delta=lines_delta, replanned=False)

    def _lost_piece(self, state: GameState, piece_type: str) -> bool:
        return state.falling is None or state.falling.name != piece_type or state.game_over

    def _locked_result(self, misexec: int, lines_before: int, replanned: bool) -> ExecResult:
        lines_delta = self._await_spawn(lines_before)
        return ExecResult(locked=True, misexec=misexec, lines_delta=lines_delta, replanned=replanned)

    def _await_spawn(self, lines_before: int) -> int:
        """Wait out the lock/line-clear animation until the next piece appears."""
        for _ in range(_MAX_SPAWN_WAIT):
            state = read_state(self.emu)
            if state.game_over or state.falling is not None:
                break
            self.emu.tick(2)
        return self.emu.lines - lines_before
