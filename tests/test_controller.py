import numpy as np
import pytest

from tetris_agent.board import BLANK
from tetris_agent.controller import Controller
from tetris_agent.pieces import PIECES, SHAPES, shape_width
from tetris_agent.policy import Placement


class FakeEmulator:
    """Minimal scripted stand-in for Emulator: J piece falling on an empty board."""

    def __init__(self, piece="J", rot=0, col=4, wall_col=0, rotation_fails=0, locks_after=3):
        self.piece = piece
        self.rot = rot
        self.col = col
        self.wall_col = wall_col
        self.rotation_fails = rotation_fails
        self.locks_after = locks_after
        self.down_presses = 0
        self.locked_cells: set[tuple[int, int]] = set()
        self.lines = 0
        self.score = 0
        self.level = 0
        self.game_over = False

    def _cells(self):
        return frozenset((1 + r, self.col + c) for r, c in SHAPES[(self.piece, self.rot)])

    def read(self, addr):
        return PIECES[self.piece] | self.rot

    def falling_sprite_cells(self):
        return self._cells()

    def game_area(self):
        area = np.full((18, 10), BLANK, dtype=int)
        for r, c in self.locked_cells:
            area[r, c] = 129
        for r, c in self._cells():
            area[r, c] = 129
        return area

    def tick(self, n=1):
        pass

    def press(self, button, ticks_after=4):
        if button == "a":
            if self.rotation_fails > 0:
                self.rotation_fails -= 1
            else:
                self.rot = (self.rot - 1) % 4
        elif button == "left":
            if self.col > self.wall_col:
                self.col -= 1
        elif button == "right":
            if self.col + shape_width(SHAPES[(self.piece, self.rot)]) < 10:
                self.col += 1
        elif button == "down":
            self.down_presses += 1
            if self.down_presses >= self.locks_after:
                # Lock at the bottom and respawn at the top.
                shape = SHAPES[(self.piece, self.rot)]
                base = 17 - max(r for r, _ in shape)
                for r, c in shape:
                    self.locked_cells.add((base + r, self.col + c))
                self.rot, self.col, self.down_presses = 0, 4, 0


def test_rotation_converges_without_misexec():
    emu = FakeEmulator(rot=0)
    result = Controller(emu).execute(Placement(rotation=2, col=4, score=0.0))
    assert result.locked
    assert result.misexec == 0


def test_unverified_rotation_counts_misexec():
    emu = FakeEmulator(rot=0, rotation_fails=1)
    result = Controller(emu).execute(Placement(rotation=3, col=4, score=0.0))
    assert result.locked
    assert result.misexec == 1
    from tetris_agent.state import normalize_cells

    assert normalize_cells(emu.locked_cells) == SHAPES[("J", 3)]


def test_shift_gives_up_at_wall_and_flags_replan():
    emu = FakeEmulator(col=4, wall_col=2)
    result = Controller(emu).execute(Placement(rotation=0, col=0, score=0.0))
    assert result.replanned
    assert result.misexec >= 3
    # The piece locked as far left as the wall allowed (fake respawns col to 4 after lock).
    assert min(c for _, c in emu.locked_cells) == 2


@pytest.mark.rom
def test_execute_locks_a_real_piece(rom_path):
    from tetris_agent.emulator import Emulator
    from tetris_agent.policy import Genome, plan_placement
    from tetris_agent.state import read_state

    emu = Emulator(rom_path)
    try:
        emu.start(timer_div=0x00)
        state = read_state(emu)
        assert state.falling is not None
        before = int(state.board.sum())
        placement = plan_placement(state.board, state.falling.name, Genome())
        result = Controller(emu, ticks_per_press=4).execute(placement)
        assert result.locked
        after_state = read_state(emu)
        assert int(after_state.board.sum()) == before + 4
        assert result.misexec <= 2
    finally:
        emu.stop()
